# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
# Modified by ComfyUI-GGUF-Next: extra layer types, quantized embedding row-gather, dequant-on-copy at load
import gguf
import torch
import logging
import threading
import contextlib

import comfy.ops
import comfy.lora
import comfy.model_management
from .dequant import dequantize_tensor, dequantize, is_quantized

def chained_hasattr(obj, chained_attr):
    probe = obj
    for attr in chained_attr.split('.'):
        if hasattr(probe, attr):
            probe = getattr(probe, attr)
        else:
            return False
    return True

# A bakcward and forward compatible way to get `torch.compiler.disable`.
def get_torch_compiler_disable_decorator():
    def dummy_decorator(*args, **kwargs):
        def noop(x):
            return x
        return noop

    from packaging import version

    if not chained_hasattr(torch, "compiler.disable"):
        logging.info("ComfyUI-GGUF: Torch too old for torch.compile - bypassing")
        return dummy_decorator # torch too old
    elif version.parse(torch.__version__) >= version.parse("2.8"):
        logging.info("ComfyUI-GGUF: Allowing full torch compile")
        return dummy_decorator # torch compile works
    if chained_hasattr(torch, "_dynamo.config.nontraceable_tensor_subclasses"):
        logging.info("ComfyUI-GGUF: Allowing full torch compile (nightly)")
        return dummy_decorator # torch compile works, nightly before 2.8 release
    else:
        logging.info("ComfyUI-GGUF: Partial torch compile only, consider updating pytorch")
        return torch.compiler.disable

torch_compiler_disable = get_torch_compiler_disable_decorator()

# `comfy.ops.cast_to` is scheduled for removal upstream, prefer model_management
_cast_to = getattr(comfy.model_management, "cast_to", None) or comfy.ops.cast_to

_TENSOR_COPY = torch.Tensor.copy_

class GGMLTensor(torch.Tensor):
    """
    Main tensor-like class for storing quantized weights
    """
    def __init__(self, *args, tensor_type, tensor_shape, patches=[], **kwargs):
        super().__init__()
        self.tensor_type = tensor_type
        self.tensor_shape = tensor_shape
        self.patches = patches

    def __new__(cls, *args, tensor_type, tensor_shape, patches=[], **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def to(self, *args, **kwargs):
        new = super().to(*args, **kwargs)
        new.tensor_type = getattr(self, "tensor_type", None)
        new.tensor_shape = getattr(self, "tensor_shape", new.data.shape)
        new.patches = getattr(self, "patches", []).copy()
        return new

    def clone(self, *args, **kwargs):
        return self

    def detach(self, *args, **kwargs):
        return self

    def copy_(self, *args, **kwargs):
        # fixes .weight.copy_ in comfy/clip_model/CLIPTextModel
        try:
            return super().copy_(*args, **kwargs)
        except Exception as e:
            logging.warning(f"ignoring 'copy_' on tensor: {e}")

    def new_empty(self, size, *args, **kwargs):
        # Intel Arc fix, ref#50
        new_tensor = super().new_empty(size, *args, **kwargs)
        return GGMLTensor(
                new_tensor,
                tensor_type = getattr(self, "tensor_type", None),
                tensor_shape = size,
                patches = getattr(self, "patches", []).copy()
        )

    @property
    def shape(self):
        if not hasattr(self, "tensor_shape"):
            self.tensor_shape = self.size()
        return self.tensor_shape

def _dequant_on_copy_torch_function(cls, func, types, args=(), kwargs=None):
    # A quantized tensor being copied into a regular parameter/buffer
    # (i.e. a module that isn't one of our GGML layers, like a bare
    # nn.Parameter `scale_shift_table` or `modulation`). Copying raw
    # quant blocks would silently corrupt it, so dequantize first.
    if func is _TENSOR_COPY and len(args) >= 2:
        dst, src = args[0], args[1]
        if isinstance(src, GGMLTensor) and not isinstance(dst, GGMLTensor) and is_quantized(src):
            src = dequantize_tensor(src, dtype=dst.dtype)
            if isinstance(src, GGMLTensor):
                src = src.as_subclass(torch.Tensor)
            with torch._C.DisableTorchFunctionSubclass():
                return _TENSOR_COPY(dst, src, *args[2:], **(kwargs or {}))
    return _BASE_TORCH_FUNCTION(cls, func, types, args, kwargs)

_BASE_TORCH_FUNCTION = torch.Tensor.__torch_function__.__func__
_load_ctx = {"depth": 0, "lock": threading.Lock()}

@contextlib.contextmanager
def dequant_on_copy():
    """
    Active only while a state dict is being loaded into a model. Kept out of
    inference on purpose: a custom __torch_function__ on GGMLTensor breaks
    torch.compile (TorchCompileModel) of the whole model.
    """
    with _load_ctx["lock"]:
        if _load_ctx["depth"] == 0:
            GGMLTensor.__torch_function__ = classmethod(_dequant_on_copy_torch_function)
        _load_ctx["depth"] += 1
    try:
        yield
    finally:
        with _load_ctx["lock"]:
            _load_ctx["depth"] -= 1
            if _load_ctx["depth"] == 0:
                del GGMLTensor.__torch_function__

class GGMLLayer(torch.nn.Module):
    """
    This (should) be responsible for de-quantizing on the fly
    """
    comfy_cast_weights = True
    dequant_dtype = None
    patch_dtype = None
    largest_layer = False
    torch_compatible_tensor_types = {None, gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}

    def is_ggml_quantized(self, *, weight=None, bias=None):
        if weight is None:
            weight = getattr(self, "weight", None)
        if bias is None:
            bias = getattr(self, "bias", None)
        return is_quantized(weight) or is_quantized(bias)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        weight, bias = state_dict.get(f"{prefix}weight"), state_dict.get(f"{prefix}bias")
        # NOTE: using modified load for linear due to not initializing on creation, see GGMLOps todo
        if self.is_ggml_quantized(weight=weight, bias=bias) or isinstance(self, torch.nn.Linear):
            return self.ggml_load_from_state_dict(state_dict, prefix, *args, **kwargs)
        # Not strictly required, but fixes embedding shape mismatch. Threshold set in loader.py
        if isinstance(self, torch.nn.Embedding) and self.weight.shape[0] >= (64 * 1024):
            return self.ggml_load_from_state_dict(state_dict, prefix, *args, **kwargs)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def ggml_load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        prefix_len = len(prefix)
        for k,v in state_dict.items():
            if k[prefix_len:] == "weight":
                self.weight = torch.nn.Parameter(v, requires_grad=False)
            elif k[prefix_len:] == "bias" and v is not None:
                self.bias = torch.nn.Parameter(v, requires_grad=False)
            else:
                unexpected_keys.append(k)

        # For Linear layer with missing weight
        if self.weight is None and isinstance(self, torch.nn.Linear):
            v = torch.zeros(self.in_features, self.out_features)
            self.weight = torch.nn.Parameter(v, requires_grad=False)
            missing_keys.append(prefix+"weight")

        # for vram estimation (TODO: less fragile logic?)
        if getattr(self.weight, "is_largest_weight", False):
            self.largest_layer = True

    def _save_to_state_dict(self, *args, **kwargs):
        if self.is_ggml_quantized():
            return self.ggml_save_to_state_dict(*args, **kwargs)
        return super()._save_to_state_dict(*args, **kwargs)

    def ggml_save_to_state_dict(self, destination, prefix, keep_vars):
        # This is a fake state dict for vram estimation
        weight = torch.zeros_like(self.weight, device=torch.device("meta"))
        destination[prefix + "weight"] = weight
        if self.bias is not None:
            bias = torch.zeros_like(self.bias, device=torch.device("meta"))
            destination[prefix + "bias"] = bias

        # Take into account space required for dequantizing the largest tensor
        if self.largest_layer:
            shape = getattr(self.weight, "tensor_shape", self.weight.shape)
            dtype = self.dequant_dtype if self.dequant_dtype and self.dequant_dtype != "target" else torch.float16
            temp = torch.empty(*shape, device=torch.device("meta"), dtype=dtype)
            destination[prefix + "temp.weight"] = temp

    def get_weight(self, tensor, dtype):
        if tensor is None:
            return

        # consolidate and load patches to GPU in async
        patch_list = []
        device = tensor.device
        key = None
        for patches, key in getattr(tensor, "patches", []):
            patch_list += move_patch_to_device(patches, device)

        # dequantize tensor while patches load
        weight = dequantize_tensor(tensor, dtype, self.dequant_dtype)

        # prevent propagating custom tensor class
        if isinstance(weight, GGMLTensor):
            weight = torch.Tensor(weight) # NOTE: as_subclass() breaks torch.compile here

        # apply patches
        if len(patch_list) > 0:
            if self.patch_dtype is None:
                weight = comfy.lora.calculate_weight(patch_list, weight, key)
            else:
                # for testing, may degrade image quality
                patch_dtype = dtype if self.patch_dtype == "target" else self.patch_dtype
                weight = comfy.lora.calculate_weight(patch_list, weight, key, patch_dtype)
        return weight

    @torch_compiler_disable()
    def cast_bias_weight(s, input=None, dtype=None, device=None, bias_dtype=None):
        if input is not None:
            if dtype is None:
                dtype = getattr(input, "dtype", torch.float32)
            if bias_dtype is None:
                bias_dtype = dtype
            if device is None:
                device = input.device

        bias = None
        non_blocking = comfy.model_management.device_supports_non_blocking(device)
        if s.bias is not None:
            bias = s.get_weight(s.bias.to(device), dtype)
            bias = _cast_to(bias, bias_dtype, device, non_blocking=non_blocking, copy=False)

        weight = s.get_weight(s.weight.to(device), dtype)
        weight = _cast_to(weight, dtype, device, non_blocking=non_blocking, copy=False)
        return weight, bias

    def forward_comfy_cast_weights(self, input, *args, **kwargs):
        if self.is_ggml_quantized():
            out = self.forward_ggml_cast_weights(input, *args, **kwargs)
        else:
            out = super().forward_comfy_cast_weights(input, *args, **kwargs)

        # non-ggml forward might still propagate custom tensor class
        if isinstance(out, GGMLTensor):
            out = torch.Tensor(out)
        return out

    def forward_ggml_cast_weights(self, input):
        raise NotImplementedError

class GGMLOps(comfy.ops.manual_cast):
    """
    Dequantize weights on the fly before doing the compute
    """
    class Linear(GGMLLayer, comfy.ops.manual_cast.Linear):
        def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
            torch.nn.Module.__init__(self)
            # TODO: better workaround for reserved memory spike on windows
            # Issue is with `torch.empty` still reserving the full memory for the layer
            # Windows doesn't over-commit memory so without this 24GB+ of pagefile is used
            self.in_features = in_features
            self.out_features = out_features
            self.weight = None
            self.bias = None

        def forward_ggml_cast_weights(self, input):
            weight, bias = self.cast_bias_weight(input)
            return torch.nn.functional.linear(input, weight, bias)

    class Conv1d(GGMLLayer, comfy.ops.manual_cast.Conv1d):
        def forward_ggml_cast_weights(self, input):
            weight, bias = self.cast_bias_weight(input)
            return self._conv_forward(input, weight, bias)

    class Conv2d(GGMLLayer, comfy.ops.manual_cast.Conv2d):
        def forward_ggml_cast_weights(self, input):
            weight, bias = self.cast_bias_weight(input)
            return self._conv_forward(input, weight, bias)

    class Conv3d(GGMLLayer, comfy.ops.manual_cast.Conv3d):
        def forward_ggml_cast_weights(self, input, autopad=None):
            weight, bias = self.cast_bias_weight(input)
            if autopad is None:
                return self._conv_forward(input, weight, bias)
            return self._conv_forward(input, weight, bias, autopad=autopad)

    class ConvTranspose1d(GGMLLayer, comfy.ops.manual_cast.ConvTranspose1d):
        def forward_ggml_cast_weights(self, input, output_size=None):
            output_padding = self._output_padding(
                input, output_size, self.stride, self.padding, self.kernel_size, 1, self.dilation)
            weight, bias = self.cast_bias_weight(input)
            return torch.nn.functional.conv_transpose1d(
                input, weight, bias, self.stride, self.padding, output_padding, self.groups, self.dilation)

    class ConvTranspose2d(GGMLLayer, comfy.ops.manual_cast.ConvTranspose2d):
        def forward_ggml_cast_weights(self, input, output_size=None):
            output_padding = self._output_padding(
                input, output_size, self.stride, self.padding, self.kernel_size, 2, self.dilation)
            weight, bias = self.cast_bias_weight(input)
            return torch.nn.functional.conv_transpose2d(
                input, weight, bias, self.stride, self.padding, output_padding, self.groups, self.dilation)

    class Embedding(GGMLLayer, comfy.ops.manual_cast.Embedding):
        def _can_gather_rows(self):
            # row-major quantized table without LoRA patches: dequantize only the rows we need
            w = self.weight
            if w is None or not is_quantized(w) or len(getattr(w, "patches", [])) > 0:
                return False
            if self.max_norm is not None:
                return False
            shape = getattr(w, "tensor_shape", None)
            return shape is not None and len(shape) == 2 and w.data.ndim == 2 and w.data.shape[0] == shape[0]

        def _gather_rows(self, input, dtype):
            w = self.weight
            idx = input.reshape(-1).to(w.device)
            rows = torch.Tensor(w.data[idx])
            rows = rows.to(input.device)
            # same semantics as dequantize_tensor(): None keeps the kernel's native precision
            deq_dtype = dtype if self.dequant_dtype == "target" else self.dequant_dtype
            out = dequantize(rows, w.tensor_type, (idx.shape[0], w.tensor_shape[1]), dtype=deq_dtype)
            return out.to(dtype).reshape(*input.shape, w.tensor_shape[1])

        def forward_ggml_cast_weights(self, input, out_dtype=None):
            output_dtype = out_dtype
            if self.weight.dtype == torch.float16 or self.weight.dtype == torch.bfloat16:
                out_dtype = None
            if self._can_gather_rows():
                # same default as upstream: cast_bias_weight(self, ...) falls back to float32
                return self._gather_rows(input, out_dtype or torch.float32).to(dtype=output_dtype)
            weight, _bias = self.cast_bias_weight(self, device=input.device, dtype=out_dtype)
            return torch.nn.functional.embedding(
                input, weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse
            ).to(dtype=output_dtype)

        def host_rows(self, input, out_dtype=None):
            if self._can_gather_rows():
                return self._gather_rows(input, out_dtype or torch.float32)
            return super().host_rows(input, out_dtype=out_dtype)

        def _host_table(self, input):
            # a large plain table that stays on the host (e.g. a 262k-row token embedding kept in fp16):
            # copy only the looked-up rows to the device instead of the whole table on every call
            w = self.weight
            if w is None or is_quantized(w) or w.ndim != 2 or w.shape[0] < 64 * 1024:
                return False
            if w.device.type != "cpu" or input.device.type == "cpu" or self.max_norm is not None:
                return False
            if len(getattr(self, "weight_function", [])) > 0 or getattr(self, "weight_lowvram_function", None) is not None:
                return False
            return len(getattr(w, "patches", [])) == 0

        def forward_comfy_cast_weights(self, input, out_dtype=None):
            if self._host_table(input):
                # the full-table path casts the weight only when it isn't fp16/bf16, then casts the output;
                # a cast commutes with row selection, so the result is the same bit for bit
                w = torch.Tensor(self.weight)
                if out_dtype is not None and w.dtype not in (torch.float16, torch.bfloat16):
                    w_dtype = out_dtype
                else:
                    w_dtype = w.dtype
                rows = w[input.reshape(-1).cpu()].to(input.device).to(w_dtype)
                return rows.view(*input.shape, rows.shape[-1]).to(dtype=out_dtype)
            return super().forward_comfy_cast_weights(input, out_dtype=out_dtype)

    class LayerNorm(GGMLLayer, comfy.ops.manual_cast.LayerNorm):
        def forward_ggml_cast_weights(self, input):
            if self.weight is None:
                return super().forward_comfy_cast_weights(input)
            weight, bias = self.cast_bias_weight(input)
            return torch.nn.functional.layer_norm(input, self.normalized_shape, weight, bias, self.eps)

    if hasattr(comfy.ops.manual_cast, "RMSNorm"): # newer ComfyUI only
        class RMSNorm(GGMLLayer, comfy.ops.manual_cast.RMSNorm):
            def forward_ggml_cast_weights(self, input):
                if self.weight is None:
                    return super().forward_comfy_cast_weights(input)
                weight, _bias = self.cast_bias_weight(input)
                return torch.nn.functional.rms_norm(input, self.normalized_shape, weight, self.eps)

    class GroupNorm(GGMLLayer, comfy.ops.manual_cast.GroupNorm):
        def forward_ggml_cast_weights(self, input):
            weight, bias = self.cast_bias_weight(input)
            return torch.nn.functional.group_norm(input, self.num_groups, weight, bias, self.eps)

def move_patch_to_device(item, device):
    if isinstance(item, torch.Tensor):
        return item.to(device, non_blocking=True)
    elif isinstance(item, tuple):
        return tuple(move_patch_to_device(x, device) for x in item)
    elif isinstance(item, list):
        return [move_patch_to_device(x, device) for x in item]
    else:
        return item
