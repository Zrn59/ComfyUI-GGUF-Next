# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
# Modified by ComfyUI-GGUF-Next: open architecture detection, more text encoders, comfy-layout passthrough
import warnings
import logging
import torch
import gguf
import re
import os

from .ops import GGMLTensor
from .dequant import is_quantized, dequantize_tensor, TORCH_NATIVE_DTYPES, INTEGER_QTYPES

# Architectures we know load correctly. Diffusion models are detected by ComfyUI
# itself from the tensor names, so unknown arch strings are accepted with a warning
# instead of being rejected outright (this is what makes new models "just work").
IMG_ARCH_LIST = {
    "flux", "flux2", "flux3", "chroma", "chroma_radiance", "sd1", "sd2", "sdxl", "sd3", "aura",
    "hidream", "hidream_o1", "cosmos", "cosmos_predict2", "anima", "ltxv", "ltxav", "hyvid",
    "hunyuan_video", "hunyuan_video15", "hunyuan_image", "hydit", "wan", "wan2.1", "wan22",
    "lumina2", "zimage", "ming_image", "qwen_image", "qwen_image21", "ideogram4", "krea2",
    "minimax", "minimax_h3", "mochi", "pixart", "cogvideox", "kandinsky5", "omnigen2", "boogu",
    "longcat_image", "ernie", "ace", "ace15", "lens", "pixeldit", "mage_flow", "moss_image",
    "joyimage", "seedvr2", "sensenova", "trellis2", "hunyuan3d", "stable_audio", "stable_cascade",
    "comfy_dit", # generic arch written by `convert.py --arch comfy_dit`
}
# llama.cpp text architectures with a known llama.cpp -> HF key mapping
LLAMA_LIKE_ARCHS = {
    "llama", "mistral", "mistral3", "qwen2", "qwen2vl", "qwen25vl", "qwen3", "qwen3vl",
    "gemma2", "gemma3",
}
# archs whose Q/K weights were permuted by llama.cpp's LlamaModel converter
PERMUTED_QK_ARCHS = {"llama", "mistral", "mistral3"}
GEMMA_ARCHS = {"gemma", "gemma2", "gemma3", "gemma3n", "gemma4"}
TXT_ARCH_LIST = {"t5", "t5encoder"} | LLAMA_LIKE_ARCHS | {"gemma4", "qwen35", "qwen3next", "gpt-oss", "t5gemma"}
VIS_TYPE_LIST = {"clip-vision", "mmproj"}

# llama.cpp-style tensor names - if none of these are present the file already
# uses ComfyUI/HF state dict names and must not be remapped.
LCPP_KEY_MARKERS = ("blk.", "token_embd.", "output_norm.", "enc.blk.", "v.blk.", "dec.blk.")

def get_orig_shape(reader, tensor_name):
    field_key = f"comfy.gguf.orig_shape.{tensor_name}"
    field = reader.get_field(field_key)
    if field is None:
        return None
    # Has original shape metadata, so we try to decode it.
    if len(field.types) != 2 or field.types[0] != gguf.GGUFValueType.ARRAY or field.types[1] != gguf.GGUFValueType.INT32:
        raise TypeError(f"Bad original shape metadata for {field_key}: Expected ARRAY of INT32, got {field.types}")
    return torch.Size(tuple(int(field.parts[part_idx][0]) for part_idx in field.data))

def get_field(reader, field_name, field_type):
    field = reader.get_field(field_name)
    if field is None:
        return None
    elif field_type == str:
        # extra check here as this is used for checking arch string
        if len(field.types) != 1 or field.types[0] != gguf.GGUFValueType.STRING:
            raise TypeError(f"Bad type for GGUF {field_name} key: expected string, got {field.types!r}")
        return str(field.parts[field.data[-1]], encoding="utf-8")
    elif field_type in [int, float, bool]:
        return field_type(field.parts[field.data[-1]].item())
    else:
        raise TypeError(f"Unknown field type {field_type}")

def get_list_field(reader, field_name, field_type):
    field = reader.get_field(field_name)
    if field is None:
        return None
    elif field_type == str:
        return tuple(str(field.parts[part_idx], encoding="utf-8") for part_idx in field.data)
    elif field_type in [int, float, bool]:
        return tuple(field_type(field.parts[part_idx][0]) for part_idx in field.data)
    else:
        raise TypeError(f"Unknown field type {field_type}")

_SCALAR_TYPES = {
    gguf.GGUFValueType.UINT8: int, gguf.GGUFValueType.INT8: int,
    gguf.GGUFValueType.UINT16: int, gguf.GGUFValueType.INT16: int,
    gguf.GGUFValueType.UINT32: int, gguf.GGUFValueType.INT32: int,
    gguf.GGUFValueType.UINT64: int, gguf.GGUFValueType.INT64: int,
    gguf.GGUFValueType.FLOAT32: float, gguf.GGUFValueType.FLOAT64: float,
    gguf.GGUFValueType.BOOL: bool,
}

def get_gguf_metadata(reader):
    """Extract all simple metadata fields like safetensors"""
    metadata = {}
    for field_name in reader.fields:
        try:
            field = reader.get_field(field_name)
            if len(field.types) != 1:  # Simple scalar fields only
                continue
            ftype = field.types[0]
            if ftype == gguf.GGUFValueType.STRING:
                metadata[field_name] = str(field.parts[field.data[-1]], "utf-8")
            elif ftype in _SCALAR_TYPES:
                metadata[field_name] = _SCALAR_TYPES[ftype](field.parts[field.data[-1]].item())
        except Exception:
            continue
    return metadata

def uses_lcpp_names(keys):
    return any(k.startswith(LCPP_KEY_MARKERS) or ".blk." in k for k in keys)

def gguf_sd_loader(path, handle_prefix="model.diffusion_model.", is_text_model=False):
    """
    Read state dict as fake tensors
    """
    reader = gguf.GGUFReader(path)

    # filter and strip prefix
    has_prefix = False
    if handle_prefix is not None:
        prefix_len = len(handle_prefix)
        tensor_names = set(tensor.name for tensor in reader.tensors)
        has_prefix = any(s.startswith(handle_prefix) for s in tensor_names)

    tensors = []
    for tensor in reader.tensors:
        sd_key = tensor_name = tensor.name
        if has_prefix:
            if not tensor_name.startswith(handle_prefix):
                continue
            sd_key = tensor_name[prefix_len:]
        tensors.append((sd_key, tensor))

    # detect and verify architecture
    compat = None
    arch_str = get_field(reader, "general.architecture", str)
    type_str = get_field(reader, "general.type", str)
    layout = get_field(reader, "comfy.gguf.tensor_name_layout", str)
    if arch_str in [None, "pig", "cow"]:
        if is_text_model:
            if arch_str is None and not uses_lcpp_names(k for k, _ in tensors):
                # raw ComfyUI/HF state dict stored as gguf (e.g. quantized CLIP-L/G, T5)
                compat = "raw"
                arch_str = "comfy_raw"
            else:
                raise ValueError(f"This gguf file is incompatible with llama.cpp!\nConsider using safetensors or a compatible gguf file\n({path})")
        else:
            compat = "sd.cpp" if arch_str is None else arch_str
            # import here to avoid changes to convert.py breaking regular models
            from .tools.convert import detect_arch
            try:
                arch_str = detect_arch(set(val[0] for val in tensors)).arch
            except Exception as e:
                # let ComfyUI's own model detection have a go at it
                logging.warning(f"ComfyUI-GGUF: architecture not detected from keys ({e}), deferring to ComfyUI model detection")
                arch_str = "unknown"
    elif is_text_model:
        if arch_str not in TXT_ARCH_LIST and type_str not in VIS_TYPE_LIST:
            if layout == "comfy_state_dict" or not uses_lcpp_names(k for k, _ in tensors):
                logging.warning(f"ComfyUI-GGUF: unknown text architecture {arch_str!r}, loading tensor names as-is")
            else:
                raise ValueError(f"Unexpected text model architecture type in GGUF file: {arch_str!r}")
    else:
        if arch_str in TXT_ARCH_LIST and arch_str not in IMG_ARCH_LIST:
            raise ValueError(f"'{os.path.basename(path)}' is a text encoder ({arch_str}), load it with a CLIP loader instead!")
        if arch_str not in IMG_ARCH_LIST:
            logging.warning(f"ComfyUI-GGUF: unknown architecture {arch_str!r}, deferring to ComfyUI model detection")

    if compat:
        logging.warning(f"Warning: This gguf model file is loaded in compatibility mode '{compat}' [arch:{arch_str}]")

    # main loading loop
    state_dict = {}
    qtype_dict = {}
    for sd_key, tensor in tensors:
        tensor_name = tensor.name
        # torch_tensor = torch.from_numpy(tensor.data) # mmap

        # NOTE: line above replaced with this block to avoid persistent numpy warning about mmap
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
            torch_tensor = torch.from_numpy(tensor.data) # mmap

        shape = get_orig_shape(reader, tensor_name)
        if shape is None:
            shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))
            # Workaround for stable-diffusion.cpp SDXL detection.
            if compat == "sd.cpp" and arch_str == "sdxl":
                if any([tensor_name.endswith(x) for x in (".proj_in.weight", ".proj_out.weight")]):
                    while len(shape) > 2 and shape[-1] == 1:
                        shape = shape[:-1]

        # add to state dict
        if tensor.tensor_type in INTEGER_QTYPES:
            # integer payloads (tokenizer blobs, index tables) - plain tensors, never dequantized
            state_dict[sd_key] = torch_tensor.view(TORCH_NATIVE_DTYPES[tensor.tensor_type]).view(*shape)
        else:
            if tensor.tensor_type in TORCH_NATIVE_DTYPES:
                torch_tensor = torch_tensor.view(TORCH_NATIVE_DTYPES[tensor.tensor_type]).view(*shape)
            state_dict[sd_key] = GGMLTensor(torch_tensor, tensor_type=tensor.tensor_type, tensor_shape=shape)

            # 1D tensors shouldn't be quantized - this is a fix for BF16 and for
            # converters (sd.cpp & co) that quantize 1D norm weights / scalars
            if len(shape) <= 1 and is_quantized(state_dict[sd_key]):
                state_dict[sd_key] = dequantize_tensor(state_dict[sd_key], dtype=torch.float32)

        # keep track of loaded tensor types
        tensor_type_str = getattr(tensor.tensor_type, "name", repr(tensor.tensor_type))
        qtype_dict[tensor_type_str] = qtype_dict.get(tensor_type_str, 0) + 1

    # print loaded tensor type counts
    logging.info("gguf qtypes: " + ", ".join(f"{k} ({v})" for k, v in qtype_dict.items()))

    # mark largest tensor for vram estimation
    qsd = {k:v for k,v in state_dict.items() if is_quantized(v)}
    if len(qsd) > 0:
        max_key = max(qsd.keys(), key=lambda k: qsd[k].numel())
        state_dict[max_key].is_largest_weight = True

    # extra info to return
    extra = {
        "arch_str": arch_str,
        "compat": compat,
        "qtypes": qtype_dict,
        "metadata": get_gguf_metadata(reader)
    }
    return (state_dict, extra)

# for remapping llama.cpp -> original key names
T5_SD_MAP = {
    "enc.": "encoder.",
    ".blk.": ".block.",
    "token_embd": "shared",
    "output_norm": "final_layer_norm",
    "attn_q": "layer.0.SelfAttention.q",
    "attn_k": "layer.0.SelfAttention.k",
    "attn_v": "layer.0.SelfAttention.v",
    "attn_o": "layer.0.SelfAttention.o",
    "attn_norm": "layer.0.layer_norm",
    "attn_rel_b": "layer.0.SelfAttention.relative_attention_bias",
    "ffn_up": "layer.1.DenseReluDense.wi_1",
    "ffn_down": "layer.1.DenseReluDense.wo",
    "ffn_gate": "layer.1.DenseReluDense.wi_0",
    "ffn_norm": "layer.1.layer_norm",
}

LLAMA_SD_MAP = {
    "blk.": "model.layers.",
    "attn_norm": "input_layernorm",
    "attn_q_norm.": "self_attn.q_norm.",
    "attn_k_norm.": "self_attn.k_norm.",
    "attn_v_norm.": "self_attn.v_norm.",
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
    "ffn_gate": "mlp.gate_proj",
    "ffn_norm": "post_attention_layernorm",
    "token_embd": "model.embed_tokens",
    "output_norm": "model.norm",
    "output.weight": "lm_head.weight",
}

GEMMA3_SD_MAP = LLAMA_SD_MAP.copy()
GEMMA3_SD_MAP.update({
    "ffn_norm": "pre_feedforward_layernorm",
    "post_ffw_norm": "post_feedforward_layernorm",
    "post_attention_norm": "post_attention_layernorm",
})
# Gemma2 shares the 4-norm sandwich layout with Gemma3 (minus q/k norm)
GEMMA2_SD_MAP = GEMMA3_SD_MAP

CLIP_VISION_SD_MAP = {
    "mm.": "visual.merger.mlp.",
    "v.post_ln.": "visual.merger.ln_q.",
    "v.patch_embd": "visual.patch_embed.proj",
    "v.blk.": "visual.blocks.",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
    "ffn_gate": "mlp.gate_proj",
    "attn_out.": "attn.proj.",
    "ln1.": "norm1.",
    "ln2.": "norm2.",
}

QWEN3_VISION_SD_MAP = {
    "mm.0.": "visual.merger.linear_fc1.",
    "mm.2.": "visual.merger.linear_fc2.",
    "v.post_ln.": "visual.merger.norm.",
    "v.position_embd.": "visual.pos_embed.",
    "v.patch_embd": "visual.patch_embed.proj",
    "v.blk.": "visual.blocks.",
    "ffn_up": "mlp.linear_fc1",
    "ffn_down": "mlp.linear_fc2",
    "attn_qkv.": "attn.qkv.",
    "attn_out.": "attn.proj.",
    "ln1.": "norm1.",
    "ln2.": "norm2.",
}

def sd_map_replace(raw_sd, key_map):
    sd = {}
    for k,v in raw_sd.items():
        for s,d in key_map.items():
            k = k.replace(s,d)
        sd[k] = v
    return sd

def llama_permute(raw_sd, n_head, n_head_kv):
    # Reverse version of LlamaModel.permute in llama.cpp convert script
    # NOTE: works directly on quantized data too - rows are independent blocks
    sd = {}
    permute = lambda x,h: x.reshape(h, x.shape[0] // h // 2, 2, *x.shape[1:]).swapaxes(1, 2).reshape(x.shape)
    for k,v in raw_sd.items():
        if k.endswith(("q_proj.weight", "q_proj.bias")):
            v.data = permute(v.data, n_head)
        if k.endswith(("k_proj.weight", "k_proj.bias")):
            v.data = permute(v.data, n_head_kv)
        sd[k] = v
    return sd

def gemma_norm_corrections(sd):
    # Reverse change from Gemma{2,3}Model modify_tensors in llama.cpp convert script (norm + 1)
    norm_patterns = [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "pre_feedforward_layernorm.weight",
        "post_feedforward_layernorm.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
        "model.norm.weight"
    ]
    for key in list(sd.keys()):
        if any(p in key for p in norm_patterns):
            if is_quantized(sd[key]):
                sd[key] = dequantize_tensor(sd[key], dtype=torch.float32) - 1.0
            else:
                sd[key] = sd[key].float() - 1.0
    return sd

gemma3_norm_corrections = gemma_norm_corrections # backwards compat alias

def strip_quant_suffix(name):
    pattern = r"[-_.]?(?:ud-)?i?q[0-9]_[a-z0-9_\-]{1,8}$"
    match = re.search(pattern, name, re.IGNORECASE)
    if match:
        name = name[:match.start()]
    for suffix in ("-f16", "-bf16", "-f32", ".f16", ".bf16", "_f16", "_bf16"):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    return name

# llama.cpp mmproj projector types accepted per text encoder arch
MMPROJ_PROJECTORS = {
    "qwen2vl": {"qwen2vl_merger", "qwen2.5vl_merger"},
    "qwen25vl": {"qwen2vl_merger", "qwen2.5vl_merger"},
    "qwen3vl": {"qwen3vl_merger"},
}

def _mmproj_projector(path):
    try:
        return get_field(gguf.GGUFReader(path), "clip.projector_type", str)
    except Exception:
        return None

def find_mmproj(path, expected_projectors=None):
    # get name to match w/o quant suffix
    tenc_fname = os.path.basename(path)
    tenc = os.path.splitext(tenc_fname)[0].lower()
    tenc = strip_quant_suffix(tenc)

    # try and find matching mmproj
    candidates = []
    target = []
    root = os.path.dirname(path)
    for fname in sorted(os.listdir(root)):
        name, ext = os.path.splitext(fname)
        if ext.lower() != ".gguf" or "mmproj" not in name.lower():
            continue
        candidates.append(fname)
        if tenc in name.lower():
            target.append(fname)

    # names don't line up (renamed files): fall back to the only mmproj in the folder,
    # but only if it is the right kind of projector for this text encoder
    if len(target) == 0 and len(candidates) == 1:
        projector = _mmproj_projector(os.path.join(root, candidates[0]))
        if expected_projectors is None or projector in expected_projectors:
            target = candidates
        else:
            logging.warning(f"Ignoring '{candidates[0]}': projector type {projector!r} doesn't match this text encoder")

    if len(target) == 0:
        return None, tenc
    if len(target) > 1:
        logging.error(f"Ambiguous mmproj for text encoder '{tenc_fname}', will use first match.")
    return os.path.join(root, target[0]), tenc

def gguf_mmproj_loader(path, arch=None):
    # Reverse version of Qwen2VLVisionModel.modify_tensors
    logging.info("Attempting to find mmproj file for text encoder...")
    target, tenc = find_mmproj(path, MMPROJ_PROJECTORS.get(arch))
    if target is None:
        logging.error(f"Error: Can't find mmproj file for '{os.path.basename(path)}' (matching:'{tenc}')! Vision input (e.g. Qwen-Image-Edit) will be broken!")
        return {}

    logging.info(f"Using mmproj '{os.path.basename(target)}' for text encoder '{os.path.basename(path)}'.")
    vsd, extra = gguf_sd_loader(target, is_text_model=True)

    # concat 4D to 5D
    if "v.patch_embd.weight.1" in vsd:
        w1 = dequantize_tensor(vsd.pop("v.patch_embd.weight"), dtype=torch.float32)
        w2 = dequantize_tensor(vsd.pop("v.patch_embd.weight.1"), dtype=torch.float32)
        vsd["v.patch_embd.weight"] = torch.stack([w1, w2], dim=2)

    # run main replacement
    key_map = CLIP_VISION_SD_MAP
    if extra["metadata"].get("clip.projector_type") == "qwen3vl_merger":
        key_map = QWEN3_VISION_SD_MAP.copy()
        deepstack_layers = sorted({int(k.split(".")[2]) for k in vsd if k.startswith("v.deepstack.")})
        for i, layer in enumerate(deepstack_layers):
            for src, dst in (("fc1", "linear_fc1"), ("fc2", "linear_fc2"), ("norm", "norm")):
                key_map[f"v.deepstack.{layer}.{src}."] = f"visual.deepstack_merger_list.{i}.{dst}."
    vsd = sd_map_replace(vsd, key_map)

    # handle split Q/K/V
    if "visual.blocks.0.attn_q.weight" in vsd:
        attns = {}
        # filter out attentions + group
        for k,v in vsd.items():
            if any(x in k for x in ["attn_q", "attn_k", "attn_v"]):
                k_attn, k_name = k.rsplit(".attn_", 1)
                k_attn += ".attn.qkv." + k_name.split(".")[-1]
                if k_attn not in attns:
                    attns[k_attn] = {}
                attns[k_attn][k_name] = dequantize_tensor(
                    v, dtype=(torch.bfloat16 if is_quantized(v) else torch.float16)
                )

        # recombine
        for k,v in attns.items():
            suffix = k.split(".")[-1]
            vsd[k] = torch.cat([
                v[f"q.{suffix}"],
                v[f"k.{suffix}"],
                v[f"v.{suffix}"],
            ], dim=0)
        del attns

        # drop the now merged split tensors
        vsd = {k: v for k, v in vsd.items() if not re.search(r"\.attn_[qkv]\.", k)}

    return vsd

def gguf_tokenizer_loader(path, temb_shape):
    # convert gguf tokenizer to spiece
    logging.info("Attempting to recreate sentencepiece tokenizer from GGUF file metadata...")
    try:
        from sentencepiece import sentencepiece_model_pb2 as model
    except ImportError:
        raise ImportError("Please make sure sentencepiece and protobuf are installed.\npip install sentencepiece protobuf")
    spm = model.ModelProto()

    reader = gguf.GGUFReader(path)

    if get_field(reader, "tokenizer.ggml.model", str) == "t5":
        if temb_shape == (256384, 4096): # probably UMT5
            spm.trainer_spec.model_type = 1 # Unigram (do we have a T5 w/ BPE?)
        else:
            raise NotImplementedError("Unknown model, can't set tokenizer!")
    else:
        raise NotImplementedError("Unknown model, can't set tokenizer!")

    spm.normalizer_spec.add_dummy_prefix = get_field(reader, "tokenizer.ggml.add_space_prefix", bool)
    spm.normalizer_spec.remove_extra_whitespaces = get_field(reader, "tokenizer.ggml.remove_extra_whitespaces", bool)

    tokens = get_list_field(reader, "tokenizer.ggml.tokens", str)
    scores = get_list_field(reader, "tokenizer.ggml.scores", float)
    toktypes = get_list_field(reader, "tokenizer.ggml.token_type", int)

    for idx, (token, score, toktype) in enumerate(zip(tokens, scores, toktypes)):
        piece = spm.SentencePiece()
        piece.piece = token
        piece.score = score
        piece.type = toktype
        spm.pieces.append(piece)

    # unsure if any of these are correct
    spm.trainer_spec.byte_fallback = True
    spm.trainer_spec.vocab_size = len(tokens) # split off unused?
    spm.trainer_spec.max_sentence_length = 4096
    spm.trainer_spec.eos_id = get_field(reader, "tokenizer.ggml.eos_token_id", int)
    spm.trainer_spec.pad_id = get_field(reader, "tokenizer.ggml.padding_token_id", int)

    logging.info(f"Created tokenizer with vocab size of {len(spm.pieces)}")
    del reader
    return torch.ByteTensor(list(spm.SerializeToString()))

def gguf_tekken_tokenizer_loader(path, temb_shape):
    # convert ggml (hf) tokenizer metadata to tekken/comfy data
    logging.info("Attempting to recreate tekken tokenizer from GGUF file metadata...")
    import json
    import base64
    from transformers.convert_slow_tokenizer import bytes_to_unicode

    reader = gguf.GGUFReader(path)

    model_str = get_field(reader, "tokenizer.ggml.model", str)
    if model_str == "gpt2":
        if temb_shape[0] == 131072: # Mistral Small 3.x (5120) / Ministral 3 (3072)
            data = {
                "config": {"num_vocab_tokens": 150000, "default_vocab_size": 131072},
                "vocab": [],
                "special_tokens": [],
            }
        else:
            raise NotImplementedError("Unknown model, can't set tokenizer!")
    else:
        raise NotImplementedError("Unknown model, can't set tokenizer!")

    tokens = get_list_field(reader, "tokenizer.ggml.tokens", str)
    toktypes = get_list_field(reader, "tokenizer.ggml.token_type", int)

    decoder = {v: k for k, v in bytes_to_unicode().items()}
    for idx, (token, toktype) in enumerate(zip(tokens, toktypes)):
        if toktype == 3:
            data["special_tokens"].append(
                {'rank': idx, 'token_str': token, 'is_control': True}
            )
        else:
            tok = bytes([decoder[char] for char in token])
            data["vocab"].append({
                "rank": len(data["vocab"]),
                "token_bytes": base64.b64encode(tok).decode("ascii"),
                "token_str": tok.decode("utf-8", errors="replace") # ?
            })

    logging.info(f"Created tekken tokenizer with vocab size of {len(data['vocab'])} (+{len(data['special_tokens'])})")
    del reader
    return torch.ByteTensor(list(json.dumps(data).encode('utf-8')))

def gguf_gemma_tokenizer_loader(path):
    # Gemma 2 / Gemma 3 sentencepiece (BPE, byte fallback, identity normalizer)
    logging.info("Attempting to recreate sentencepiece tokenizer from GGUF file metadata...")
    try:
        from sentencepiece import sentencepiece_model_pb2 as model
    except ImportError:
        raise ImportError("Please install sentencepiece and protobuf.\npip install sentencepiece protobuf")
    spm = model.ModelProto()
    reader = gguf.GGUFReader(path)

    spm.normalizer_spec.name = "identity"
    spm.normalizer_spec.add_dummy_prefix = False
    spm.trainer_spec.model_type = 2
    spm.trainer_spec.input_format = "tsv"
    spm.trainer_spec.byte_fallback = True
    spm.trainer_spec.max_sentence_length = 4192
    spm.trainer_spec.bos_piece = "<bos>"

    tokens = get_list_field(reader, "tokenizer.ggml.tokens", str)
    scores = get_list_field(reader, "tokenizer.ggml.scores", float)
    toktype = get_list_field(reader, "tokenizer.ggml.token_type", int)

    if not tokens or not scores or not toktype:
        raise ValueError("Missing tokenizer metadata")

    for idx in range(len(tokens)):
        piece = spm.SentencePiece()
        piece.piece = tokens[idx]
        if idx == 3:  # UNK position
            piece.type = 2  # UNK Token
            piece.score = 0.0 # UNK Score
        else:
            piece.type = toktype[idx]
            piece.score = scores[idx]
        spm.pieces.append(piece)

    spm.trainer_spec.vocab_size = len(spm.pieces)
    logging.info(f"Created tokenizer with vocab size of {len(spm.pieces)}")

    del reader
    return torch.ByteTensor(list(spm.SerializeToString()))

gguf_gemma3_tokenizer_loader = gguf_gemma_tokenizer_loader # backwards compat alias

def _head_counts(metadata, arch, default=(32, 8)):
    n_head = metadata.get(f"{arch}.attention.head_count")
    n_head_kv = metadata.get(f"{arch}.attention.head_count_kv", n_head)
    if not n_head:
        return default
    return int(n_head), int(n_head_kv)

def gguf_clip_loader(path):
    sd, extra = gguf_sd_loader(path, is_text_model=True)
    arch = extra.get("arch_str", None)
    metadata = extra.get("metadata", {})

    # Files that already store ComfyUI/HF tensor names (e.g. custom Gemma4 / Qwen3.5
    # conversions, quantized CLIP/T5 dumps) are passed through untouched.
    layout = metadata.get("comfy.gguf.tensor_name_layout")
    if layout == "comfy_state_dict" or not uses_lcpp_names(sd.keys()):
        logging.info(f"ComfyUI-GGUF: '{os.path.basename(path)}' uses ComfyUI tensor names, skipping llama.cpp remap [arch:{arch}]")
        # Same policy as below: tied-embedding logits (generate / prompt enhance) read the
        # whole table through comfy's own cast path, so large vocab tables can't stay quantized.
        for k in list(sd.keys()):
            if k.endswith(("embed_tokens.weight", "shared.weight")) and is_quantized(sd[k]) and sd[k].shape[0] >= (64 * 1024):
                logging.warning(f"Dequantizing {k} to prevent runtime OOM.")
                sd[k] = dequantize_tensor(sd[k], dtype=torch.float16)
        return sd

    if arch in {"t5", "t5encoder"}:
        temb_key = "token_embd.weight"
        if temb_key in sd and sd[temb_key].shape == (256384, 4096):
            # non-standard Comfy-Org tokenizer
            sd["spiece_model"] = gguf_tokenizer_loader(path, sd[temb_key].shape)
            # TODO: dequantizing token embed here is janky but otherwise we OOM due to tensor being massive.
            logging.warning(f"Dequantizing {temb_key} to prevent runtime OOM.")
            sd[temb_key] = dequantize_tensor(sd[temb_key], dtype=torch.float16)
        sd = sd_map_replace(sd, T5_SD_MAP)
    elif arch in LLAMA_LIKE_ARCHS:
        # TODO: pass model_options["vocab_size"] to loader somehow
        temb_key = "token_embd.weight"
        if temb_key in sd and sd[temb_key].shape[0] >= (64 * 1024):
            if arch in PERMUTED_QK_ARCHS and sd[temb_key].shape[0] == 131072:
                # non-standard Comfy-Org tokenizer (Mistral Small 3.x / Ministral 3)
                sd["tekken_model"] = gguf_tekken_tokenizer_loader(path, sd[temb_key].shape)
            elif arch in GEMMA_ARCHS:
                sd["spiece_model"] = gguf_gemma_tokenizer_loader(path)
            # See note above for T5.
            logging.warning(f"Dequantizing {temb_key} to prevent runtime OOM.")
            sd[temb_key] = dequantize_tensor(sd[temb_key], dtype=torch.float16)

        if arch == "gemma3":
            sd = sd_map_replace(sd, GEMMA3_SD_MAP)
            sd = gemma_norm_corrections(sd)
        elif arch == "gemma2":
            sd = sd_map_replace(sd, GEMMA2_SD_MAP)
            sd = gemma_norm_corrections(sd)
        else:
            sd = sd_map_replace(sd, LLAMA_SD_MAP)

        if arch in PERMUTED_QK_ARCHS:
            n_head, n_head_kv = _head_counts(metadata, arch)
            sd = llama_permute(sd, n_head, n_head_kv) # L3 / Mistral / Ministral

        if arch in {"qwen2vl", "qwen25vl", "qwen3vl"}:
            vsd = gguf_mmproj_loader(path, arch)
            if arch == "qwen3vl":
                if not vsd:
                    raise ValueError(f"Qwen3-VL requires a matching mmproj GGUF alongside the text encoder: {path}")
                # ComfyUI expects HF layout (model.visual.*) for the 4B/8B variants and a
                # flat layout (visual.*) for the truncated 32B MiniMax-H3 encoder.
                hidden = metadata.get("qwen3vl.embedding_length")
                if hidden in (2560, 4096):
                    vsd = {f"model.{k}": v for k, v in vsd.items()}
            sd.update(vsd)
    else:
        logging.warning(f"ComfyUI-GGUF: no llama.cpp key mapping for text architecture {arch!r}, loading tensor names as-is")
    return sd
