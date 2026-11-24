# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
# Modified by ComfyUI-GGUF-Next: extended quant coverage, device constant cache, optional torch.compile.
# Kernels are ports of the gguf-py / ggml reference code (MIT, see THIRD_PARTY_NOTICES.md)
import os
import logging
import threading

import gguf
import torch
from tqdm import tqdm

Q = gguf.GGMLQuantizationType

def _qt(name):
    """Resolve a quant type that may not exist in older gguf-py releases."""
    return getattr(Q, name, None)

# Types that map 1:1 onto a torch dtype (no dequant step needed)
TORCH_NATIVE_DTYPES = {
    Q.F32: torch.float32,
    Q.F16: torch.float16,
}
for _name, _dtype in (("F64", torch.float64), ("I8", torch.int8), ("I16", torch.int16),
                      ("I32", torch.int32), ("I64", torch.int64)):
    if _qt(_name) is not None:
        TORCH_NATIVE_DTYPES[_qt(_name)] = _dtype

INTEGER_QTYPES = {q for q, d in TORCH_NATIVE_DTYPES.items() if not d.is_floating_point}
TORCH_COMPATIBLE_QTYPES = (None, *TORCH_NATIVE_DTYPES.keys())

def is_torch_compatible(tensor):
    return tensor is None or getattr(tensor, "tensor_type", None) in TORCH_COMPATIBLE_QTYPES

def is_quantized(tensor):
    return not is_torch_compatible(tensor)

# ---------------------------------------------------------------------------
# Optional torch.compile of the block dequant functions.
# Every dequant function is a chain of 5-20 tiny elementwise kernels; inductor
# fuses them into one or two, which is a large win for per-step dequantization.
# ---------------------------------------------------------------------------
_COMPILE = {"enabled": os.environ.get("GGUF_COMPILE_DEQUANT", "0").lower() in ("1", "true", "yes")}
_compiled_fns = {}
_compile_failed = set()
_compile_warm = set()

def set_compile_dequant(enabled):
    enabled = bool(enabled)
    if enabled and not hasattr(torch, "compile"):
        logging.warning("ComfyUI-GGUF: torch.compile unavailable, dequant compile disabled")
        enabled = False
    if enabled != _COMPILE["enabled"]:
        logging.info(f"ComfyUI-GGUF: torch.compile dequant {'enabled' if enabled else 'disabled'}")
    _COMPILE["enabled"] = enabled

def get_compile_dequant():
    return _COMPILE["enabled"]

_maybe_mark_dynamic = getattr(getattr(torch, "_dynamo", None), "maybe_mark_dynamic", None)

def _get_block_fn(qtype):
    fn = dequantize_functions[qtype]
    if not _COMPILE["enabled"] or qtype in _compile_failed:
        return fn
    cfn = _compiled_fns.get(qtype)
    if cfn is None:
        # NOTE: not dynamic=True. With a symbolic type_size, inductor (torch 2.11) miscompiles
        # the kernels that keep their fp16 scale at the end of the block (Q2_K, Q3_K, Q6_K, TQ*).
        # Only the block count is marked dynamic, in dequantize().
        cfn = torch.compile(fn, dynamic=None, fullgraph=False)
        _compiled_fns[qtype] = cfn
    return cfn

def dequantize_tensor(tensor, dtype=None, dequant_dtype=None):
    qtype = getattr(tensor, "tensor_type", None)
    oshape = getattr(tensor, "tensor_shape", tensor.shape)

    if qtype in TORCH_COMPATIBLE_QTYPES:
        if qtype in INTEGER_QTYPES:
            return tensor # never cast integer payloads (tokenizer blobs, indices)
        return tensor.to(dtype)
    elif qtype in dequantize_functions:
        dequant_dtype = dtype if dequant_dtype == "target" else dequant_dtype
        return dequantize(tensor.data, qtype, oshape, dtype=dequant_dtype).to(dtype)
    else:
        # this is incredibly slow
        tqdm.write(f"Falling back to numpy dequant for qtype: {getattr(qtype, 'name', repr(qtype))}")
        new = gguf.quants.dequantize(tensor.cpu().numpy(), qtype)
        return torch.from_numpy(new).to(tensor.device, dtype=dtype).reshape(oshape)

def dequantize(data, qtype, oshape, dtype=None):
    """
    Dequantize tensor back to usable shape/dtype
    """
    block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]

    rows = data.reshape(
        (-1, data.shape[-1])
    ).view(torch.uint8)

    n_blocks = rows.numel() // type_size
    blocks = rows.reshape((n_blocks, type_size))

    fn = _get_block_fn(qtype)
    if fn is not dequantize_functions[qtype] and (qtype, blocks.device) not in _compile_warm:
        # first call per device runs eagerly so lookup tables get built outside of tracing
        _compile_warm.add((qtype, blocks.device))
        fn = dequantize_functions[qtype]
    if fn is not dequantize_functions[qtype] and _maybe_mark_dynamic is not None:
        _maybe_mark_dynamic(blocks, 0) # one graph for every tensor size
    try:
        out = fn(blocks, block_size, type_size, dtype)
    except Exception as e:
        if fn is dequantize_functions[qtype]:
            raise
        # compiled path failed - permanently fall back to eager for this qtype
        _compile_failed.add(qtype)
        logging.warning(f"ComfyUI-GGUF: torch.compile dequant failed for {qtype.name}, using eager ({e})")
        out = dequantize_functions[qtype](blocks, block_size, type_size, dtype)
    return out.reshape(oshape)

# ---------------------------------------------------------------------------
# Device constant cache. The original implementation created small index /
# shift tensors with torch.tensor(..., device=cuda) on *every* call, which is
# a synchronous host->device copy per layer per step. Build them once per device.
# ---------------------------------------------------------------------------
_const_cache = {}
_const_lock = threading.Lock()

def _const(name, device, builder):
    key = (name, device)
    t = _const_cache.get(key)
    if t is None:
        with _const_lock:
            t = _const_cache.get(key)
            if t is None:
                t = builder().to(device)
                _const_cache[key] = t
    return t

def _shifts(device, values, dtype=torch.uint8):
    values = tuple(values)
    return _const(("shift", values, dtype), device, lambda: torch.tensor(values, dtype=dtype))

def _bits8(device):
    return _shifts(device, range(8))

def to_uint32(x):
    # no uint32 :(  - returns int32 holding the same bit pattern
    x = x.view(torch.uint8).to(torch.int32)
    return (x[:, 0] | x[:, 1] << 8 | x[:, 2] << 16 | x[:, 3] << 24).unsqueeze(1)

def to_uint16(x):
    x = x.view(torch.uint8).to(torch.int32)
    return (x[:, 0] | x[:, 1] << 8).unsqueeze(1)

def bytes_to_i32(x, n_words):
    """(n, 4*k) uint8 -> (n, k) int32 (little endian, bit pattern preserved)"""
    x = x.reshape((x.shape[0], n_words, 4)).to(torch.int32)
    return x[..., 0] | (x[..., 1] << 8) | (x[..., 2] << 16) | (x[..., 3] << 24)

def bytes_to_u16(x, n_words):
    """(n, 2*k) uint8 -> (n, k) int32 holding uint16 values"""
    x = x.reshape((x.shape[0], n_words, 2)).to(torch.int32)
    return x[..., 0] | (x[..., 1] << 8)

def split_block_dims(blocks, *args):
    n_max = blocks.shape[1]
    dims = list(args) + [n_max - sum(args)]
    return torch.split(blocks, dims, dim=1)

_is_compiling = getattr(getattr(torch, "compiler", None), "is_compiling", lambda: False)

def _f16(x, dtype):
    # Eager: .view(float16) directly on the strided block slice (no copy) when rows stay
    # 2-byte aligned. Under torch.compile a contiguous copy fuses much better (Q6_K: ~2x).
    if _is_compiling() or not (x.stride(-1) == 1 and x.stride(0) % 2 == 0 and x.storage_offset() % 2 == 0):
        x = x.contiguous()
    return x.view(torch.float16).to(dtype)

def _signs_from_bits(bits, dtype):
    # bit==0 -> +1, bit==1 -> -1
    return (1 - 2 * bits.to(torch.int8)).to(dtype if dtype is not None else torch.float16)

# Full weights #
def dequantize_blocks_BF16(blocks, block_size, type_size, dtype=None):
    return (blocks.view(torch.int16).to(torch.int32) << 16).view(torch.float32)

# Legacy Quants #
def dequantize_blocks_Q8_0(blocks, block_size, type_size, dtype=None):
    d, x = split_block_dims(blocks, 2)
    d = _f16(d, dtype)
    x = x.view(torch.int8)
    return (d * x)

def dequantize_blocks_Q5_1(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, m, qh, qs = split_block_dims(blocks, 2, 2, 4)
    d = _f16(d, dtype)
    m = _f16(m, dtype)
    qh = to_uint32(qh)

    qh = qh.reshape((n_blocks, 1)) >> _shifts(d.device, range(32), torch.int32).reshape(1, 32)
    ql = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> _shifts(d.device, (0, 4)).reshape(1, 1, 2, 1)
    qh = (qh & 1).to(torch.uint8)
    ql = (ql & 0x0F).reshape((n_blocks, -1))

    qs = (ql | (qh << 4))
    return (d * qs) + m

def dequantize_blocks_Q5_0(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, qh, qs = split_block_dims(blocks, 2, 4)
    d  = _f16(d, dtype)
    qh = to_uint32(qh)

    qh = qh.reshape(n_blocks, 1) >> _shifts(d.device, range(32), torch.int32).reshape(1, 32)
    ql = qs.reshape(n_blocks, -1, 1, block_size // 2) >> _shifts(d.device, (0, 4)).reshape(1, 1, 2, 1)

    qh = (qh & 1).to(torch.uint8)
    ql = (ql & 0x0F).reshape(n_blocks, -1)

    qs = (ql | (qh << 4)).to(torch.int8) - 16
    return (d * qs)

def dequantize_blocks_Q4_1(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, m, qs = split_block_dims(blocks, 2, 2)
    d = _f16(d, dtype)
    m = _f16(m, dtype)

    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> _shifts(d.device, (0, 4)).reshape(1, 1, 2, 1)
    qs = (qs & 0x0F).reshape(n_blocks, -1)

    return (d * qs) + m

def dequantize_blocks_Q4_0(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, qs = split_block_dims(blocks, 2)
    d  = _f16(d, dtype)

    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> _shifts(d.device, (0, 4)).reshape((1, 1, 2, 1))
    qs = (qs & 0x0F).reshape((n_blocks, -1)).to(torch.int8) - 8
    return (d * qs)

# 1-bit (sign) quant, d * (+1 | -1)
def dequantize_blocks_Q1_0(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, qs = split_block_dims(blocks, 2)
    d = _f16(d, dtype)
    bits = (qs.reshape((n_blocks, -1, 1)) >> _bits8(d.device).reshape((1, 1, 8))) & 1
    bits = bits.reshape((n_blocks, block_size))
    return d * (2 * bits.to(torch.int8) - 1)

# 2-bit uniform quant, (q - 1) * d with q in 0..3
def dequantize_blocks_Q2_0(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, qs = split_block_dims(blocks, 2)
    d = _f16(d, dtype)
    qs = (qs.reshape((n_blocks, -1, 1)) >> _shifts(d.device, (0, 2, 4, 6)).reshape((1, 1, 4))) & 0x03
    qs = qs.reshape((n_blocks, block_size)).to(torch.int8) - 1
    return d * qs

# OCP FP8 (e4m3fn) - stored one byte per element, maps 1:1 onto torch.float8_e4m3fn
def dequantize_blocks_F8_E4M3(blocks, block_size, type_size, dtype=None):
    out = blocks.contiguous().view(torch.float8_e4m3fn)
    return out.to(dtype if dtype is not None else torch.float16)

# K Quants #
QK_K = 256
K_SCALE_SIZE = 12

def get_scale_min(scales):
    n_blocks = scales.shape[0]
    scales = scales.view(torch.uint8)
    scales = scales.reshape((n_blocks, 3, 4))

    d, m, m_d = torch.split(scales, scales.shape[-2] // 3, dim=-2)

    sc = torch.cat([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], dim=-1)
    min = torch.cat([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], dim=-1)

    return (sc.reshape((n_blocks, 8)), min.reshape((n_blocks, 8)))

def dequantize_blocks_Q6_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    ql, qh, scales, d, = split_block_dims(blocks, QK_K // 2, QK_K // 4, QK_K // 16)

    scales = scales.view(torch.int8).to(dtype)
    d = _f16(d, dtype)
    d = (d * scales).reshape((n_blocks, QK_K // 16, 1))

    ql = ql.reshape((n_blocks, -1, 1, 64)) >> _shifts(d.device, (0, 4)).reshape((1, 1, 2, 1))
    ql = (ql & 0x0F).reshape((n_blocks, -1, 32))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> _shifts(d.device, (0, 2, 4, 6)).reshape((1, 1, 4, 1))
    qh = (qh & 0x03).reshape((n_blocks, -1, 32))
    q = (ql | (qh << 4)).to(torch.int8) - 32
    q = q.reshape((n_blocks, QK_K // 16, -1))

    return (d * q).reshape((n_blocks, QK_K))

def dequantize_blocks_Q5_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, dmin, scales, qh, qs = split_block_dims(blocks, 2, 2, K_SCALE_SIZE, QK_K // 8)

    d = _f16(d, dtype)
    dmin = _f16(dmin, dtype)

    sc, m = get_scale_min(scales)

    d = (d * sc).reshape((n_blocks, -1, 1))
    dm = (dmin * m).reshape((n_blocks, -1, 1))

    ql = qs.reshape((n_blocks, -1, 1, 32)) >> _shifts(d.device, (0, 4)).reshape((1, 1, 2, 1))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> _bits8(d.device).reshape((1, 1, 8, 1))
    ql = (ql & 0x0F).reshape((n_blocks, -1, 32))
    qh = (qh & 0x01).reshape((n_blocks, -1, 32))
    q = (ql | (qh << 4))

    return (d * q - dm).reshape((n_blocks, QK_K))

def dequantize_blocks_Q4_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, dmin, scales, qs = split_block_dims(blocks, 2, 2, K_SCALE_SIZE)
    d = _f16(d, dtype)
    dmin = _f16(dmin, dtype)

    sc, m = get_scale_min(scales)

    d = (d * sc).reshape((n_blocks, -1, 1))
    dm = (dmin * m).reshape((n_blocks, -1, 1))

    qs = qs.reshape((n_blocks, -1, 1, 32)) >> _shifts(d.device, (0, 4)).reshape((1, 1, 2, 1))
    qs = (qs & 0x0F).reshape((n_blocks, -1, 32))

    return (d * qs - dm).reshape((n_blocks, QK_K))

def dequantize_blocks_Q3_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    hmask, qs, scales, d = split_block_dims(blocks, QK_K // 8, QK_K // 4, 12)
    d = _f16(d, dtype)

    lscales, hscales = scales[:, :8], scales[:, 8:]
    lscales = lscales.reshape((n_blocks, 1, 8)) >> _shifts(d.device, (0, 4)).reshape((1, 2, 1))
    lscales = lscales.reshape((n_blocks, 16))
    hscales = hscales.reshape((n_blocks, 1, 4)) >> _shifts(d.device, (0, 2, 4, 6)).reshape((1, 4, 1))
    hscales = hscales.reshape((n_blocks, 16))
    scales = (lscales & 0x0F) | ((hscales & 0x03) << 4)
    scales = (scales.to(torch.int8) - 32)

    dl = (d * scales).reshape((n_blocks, 16, 1))

    ql = qs.reshape((n_blocks, -1, 1, 32)) >> _shifts(d.device, (0, 2, 4, 6)).reshape((1, 1, 4, 1))
    qh = hmask.reshape(n_blocks, -1, 1, 32) >> _bits8(d.device).reshape((1, 1, 8, 1))
    ql = ql.reshape((n_blocks, 16, QK_K // 16)) & 3
    qh = (qh.reshape((n_blocks, 16, QK_K // 16)) & 1) ^ 1
    q = (ql.to(torch.int8) - (qh << 2).to(torch.int8))

    return (dl * q).reshape((n_blocks, QK_K))

def dequantize_blocks_Q2_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    scales, qs, d, dmin = split_block_dims(blocks, QK_K // 16, QK_K // 4, 2)
    d = _f16(d, dtype)
    dmin = _f16(dmin, dtype)

    # (n_blocks, 16, 1)
    dl = (d * (scales & 0xF)).reshape((n_blocks, QK_K // 16, 1))
    ml = (dmin * (scales >> 4)).reshape((n_blocks, QK_K // 16, 1))

    shift = _shifts(d.device, (0, 2, 4, 6)).reshape((1, 1, 4, 1))

    qs = (qs.reshape((n_blocks, -1, 1, 32)) >> shift) & 3
    qs = qs.reshape((n_blocks, QK_K // 16, 16))
    qs = dl * qs - ml

    return qs.reshape((n_blocks, -1))

def dequantize_blocks_Q8_K(blocks, block_size, type_size, dtype=None):
    d, qs, _bsums = split_block_dims(blocks, 4, QK_K)
    d = d.contiguous().view(torch.float32)
    if dtype is not None:
        d = d.to(dtype)
    return d * qs.view(torch.int8)

# Ternary quants #
def dequantize_blocks_TQ1_0(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    qs, qh, d = split_block_dims(blocks, (QK_K - 4 * QK_K // 64) // 5, QK_K // 64)
    d = _f16(d, dtype)

    pow3 = _shifts(d.device, (1, 3, 9, 27, 81))
    qs0, qs1 = qs[:, :32], qs[:, 32:]
    # uint8 multiplication intentionally wraps (mod 256), same as the C code
    qs0 = (qs0.reshape((n_blocks, -1, 1, 32)) * pow3.reshape((1, 1, 5, 1))).reshape((n_blocks, -1))
    qs1 = (qs1.reshape((n_blocks, -1, 1, 16)) * pow3.reshape((1, 1, 5, 1))).reshape((n_blocks, -1))
    qh = (qh.reshape((n_blocks, -1, 1, 4)) * pow3[:4].reshape((1, 1, 4, 1))).reshape((n_blocks, -1))
    qs = torch.cat([qs0, qs1, qh], dim=-1).to(torch.int32)
    qs = ((qs * 3) >> 8) - 1

    return d * qs.to(torch.int8)

def dequantize_blocks_TQ2_0(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    qs, d = split_block_dims(blocks, QK_K // 4)
    d = _f16(d, dtype)

    qs = qs.reshape((n_blocks, -1, 1, 32)) >> _shifts(d.device, (0, 2, 4, 6)).reshape((1, 1, 4, 1))
    qs = (qs & 0x03).reshape((n_blocks, -1)).to(torch.int8) - 1

    return d * qs

# IQ quants #
KVALUES = torch.tensor([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], dtype=torch.int8)
# e2m1 values (doubled), shared by MXFP4 and NVFP4
KVALUES_FP4 = torch.tensor([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=torch.int8)

def _kvalues(device):
    return _const("kvalues_iq4", device, lambda: KVALUES)

def _kvalues_fp4(device):
    return _const("kvalues_fp4", device, lambda: KVALUES_FP4)

def _ksigns(device):
    return _const("ksigns", device, lambda: torch.tensor(list(gguf.quants.IQ2_XXS.ksigns), dtype=torch.int32))

def _grid(cls_name, device):
    def build():
        cls = getattr(gguf.quants, cls_name)
        cls.init_grid()
        grid = torch.from_numpy(cls.grid.copy()).reshape(cls.grid_shape)
        return grid.to(torch.float32)
    return _const(("grid", cls_name), device, build)

def _grid_lookup(cls_name, idx, dtype):
    grid = _grid(cls_name, idx.device)
    if dtype is not None and grid.dtype != dtype:
        grid = _const(("grid", cls_name, dtype), idx.device, lambda: _grid(cls_name, idx.device).to(dtype))
    return grid[idx.long()]

def dequantize_blocks_IQ4_NL(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, qs = split_block_dims(blocks, 2)
    d = _f16(d, dtype)

    qs = qs.reshape((n_blocks, -1, 1, block_size//2)) >> _shifts(d.device, (0, 4)).reshape((1, 1, 2, 1))
    qs = (qs & 0x0F).reshape((n_blocks, -1)).long()

    return d * _kvalues(d.device)[qs]

def dequantize_blocks_IQ4_XS(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, scales_h, scales_l, qs = split_block_dims(blocks, 2, 2, QK_K // 64)
    d = _f16(d, dtype)
    scales_h = to_uint16(scales_h)

    shift_a = _shifts(d.device, (0, 4))
    shift_b = _shifts(d.device, [2 * i for i in range(QK_K // 32)], torch.int32)

    scales_l = scales_l.reshape((n_blocks, -1, 1)) >> shift_a.reshape((1, 1, 2))
    scales_h = scales_h.reshape((n_blocks, -1, 1)) >> shift_b.reshape((1, -1, 1))

    scales_l = scales_l.reshape((n_blocks, -1)) & 0x0F
    scales_h = scales_h.reshape((n_blocks, -1)).to(torch.uint8) & 0x03

    scales = (scales_l | (scales_h << 4)).to(torch.int8) - 32
    dl = (d * scales.to(d.dtype)).reshape((n_blocks, -1, 1))

    qs = qs.reshape((n_blocks, -1, 1, 16)) >> shift_a.reshape((1, 1, 2, 1))
    qs = (qs.reshape((n_blocks, -1, 32)) & 0x0F).long()
    qs = _kvalues(d.device)[qs]

    return (dl * qs).reshape((n_blocks, -1))

def dequantize_blocks_IQ2_XXS(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, qs = split_block_dims(blocks, 2)
    d = _f16(d, dtype)

    qs = qs.reshape((n_blocks, QK_K // 32, 2, 4))
    grid_idx = qs[:, :, 0, :]                                # (n, 8, 4) bytes
    aux = bytes_to_i32(qs[:, :, 1, :].reshape((n_blocks, -1)), QK_K // 32)  # (n, 8)

    db = d * (0.5 + ((aux >> 28) & 0xF).to(d.dtype)) * 0.25
    db = db.reshape((n_blocks, -1, 1, 1))

    sign_idx = (aux.unsqueeze(-1) >> _shifts(d.device, (0, 7, 14, 21), torch.int32).reshape((1, 1, 4))) & 0x7F
    signs = _ksigns(d.device)[sign_idx.long()]                 # (n, 8, 4)
    signs = (signs.unsqueeze(-1) >> _bits8(d.device).to(torch.int32).reshape((1, 1, 1, 8))) & 1
    signs = _signs_from_bits(signs, d.dtype)                 # (n, 8, 4, 8)

    grid = _grid_lookup("IQ2_XXS", grid_idx, d.dtype)        # (n, 8, 4, 8)
    return (db * grid * signs).reshape((n_blocks, -1))

def dequantize_blocks_IQ2_XS(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, qs, scales = split_block_dims(blocks, 2, 2 * QK_K // 8)
    d = _f16(d, dtype)
    qs = bytes_to_u16(qs, QK_K // 8)                         # (n, 32)

    scales = scales.reshape((n_blocks, -1, 1)) >> _shifts(d.device, (0, 4)).reshape((1, 1, 2))
    scales = (scales & 0x0F).reshape((n_blocks, -1))         # (n, 16)
    db = d * (0.5 + scales.to(d.dtype)) * 0.25
    db = db.reshape((n_blocks, -1, 1, 1))

    signs = _ksigns(d.device)[(qs >> 9).long()]              # (n, 32)
    signs = (signs.unsqueeze(-1) >> _bits8(d.device).to(torch.int32).reshape((1, 1, 8))) & 1
    signs = _signs_from_bits(signs, d.dtype).reshape((n_blocks, -1, 2, 8))

    grid = _grid_lookup("IQ2_XS", qs & 511, d.dtype).reshape((n_blocks, -1, 2, 8))
    return (db * grid * signs).reshape((n_blocks, -1))

def dequantize_blocks_IQ2_S(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, qs, signs, qh, scales = split_block_dims(blocks, 2, QK_K // 8, QK_K // 8, QK_K // 32)
    d = _f16(d, dtype)

    scales = scales.reshape((n_blocks, -1, 1)) >> _shifts(d.device, (0, 4)).reshape((1, 1, 2))
    scales = (scales & 0x0F).reshape((n_blocks, -1))
    db = d * (0.5 + scales.to(d.dtype)) * 0.25
    db = db.reshape((n_blocks, -1, 1, 1))

    signs = (signs.reshape((n_blocks, -1, 1)) >> _bits8(d.device).reshape((1, 1, 8))) & 1
    signs = _signs_from_bits(signs, d.dtype).reshape((n_blocks, -1, 2, 8))

    qh = qh.reshape((n_blocks, -1, 1)) >> _shifts(d.device, (0, 2, 4, 6)).reshape((1, 1, 4))
    idx = qs.to(torch.int32) | ((qh & 0x03).to(torch.int32) << 8).reshape((n_blocks, -1))

    grid = _grid_lookup("IQ2_S", idx, d.dtype).reshape((n_blocks, -1, 2, 8))
    return (db * grid * signs).reshape((n_blocks, -1))

def dequantize_blocks_IQ3_XXS(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, qs, scales = split_block_dims(blocks, 2, QK_K // 4)
    d = _f16(d, dtype)
    scales = bytes_to_i32(scales, QK_K // 32)                # (n, 8)

    db = d * (0.5 + ((scales >> 28) & 0xF).to(d.dtype)) * 0.5
    db = db.reshape((n_blocks, -1, 1, 1))

    sign_idx = (scales.unsqueeze(-1) >> _shifts(d.device, (0, 7, 14, 21), torch.int32).reshape((1, 1, 4))) & 0x7F
    signs = _ksigns(d.device)[sign_idx.long()]
    signs = (signs.unsqueeze(-1) >> _bits8(d.device).to(torch.int32).reshape((1, 1, 1, 8))) & 1
    signs = _signs_from_bits(signs, d.dtype)                 # (n, 8, 4, 8)

    grid = _grid_lookup("IQ3_XXS", qs, d.dtype).reshape((n_blocks, -1, 4, 8))
    return (db * grid * signs).reshape((n_blocks, -1))

def dequantize_blocks_IQ3_S(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, qs, qh, signs, scales = split_block_dims(blocks, 2, QK_K // 4, QK_K // 32, QK_K // 8)
    d = _f16(d, dtype)

    scales = scales.reshape((n_blocks, -1, 1)) >> _shifts(d.device, (0, 4)).reshape((1, 1, 2))
    scales = (scales & 0x0F).reshape((n_blocks, -1))
    db = d * (1 + 2 * scales.to(d.dtype))
    db = db.reshape((n_blocks, -1, 1, 1))

    signs = (signs.reshape((n_blocks, -1, 1)) >> _bits8(d.device).reshape((1, 1, 8))) & 1
    signs = _signs_from_bits(signs, d.dtype).reshape((n_blocks, -1, 4, 8))

    qh = (qh.reshape((n_blocks, -1, 1)) >> _bits8(d.device).reshape((1, 1, 8))) & 1
    idx = qs.to(torch.int32) | (qh.to(torch.int32).reshape((n_blocks, -1)) << 8)

    grid = _grid_lookup("IQ3_S", idx, d.dtype).reshape((n_blocks, -1, 4, 8))
    return (db * grid * signs).reshape((n_blocks, -1))

IQ1_DELTA = 0.125

def dequantize_blocks_IQ1_S(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, qs, qh = split_block_dims(blocks, 2, QK_K // 8)
    d = _f16(d, dtype)
    qh = bytes_to_u16(qh, QK_K // 32)                        # (n, 8)

    dl = d * (2 * ((qh >> 12) & 7) + 1).to(d.dtype)
    dl = dl.reshape((n_blocks, -1, 1, 1))
    delta = torch.where((qh & 0x8000) == 0, IQ1_DELTA, -IQ1_DELTA).to(d.dtype)
    delta = delta.reshape((n_blocks, -1, 1, 1))

    qh = qh.reshape((n_blocks, -1, 1)) >> _shifts(d.device, (0, 3, 6, 9), torch.int32).reshape((1, 1, 4))
    idx = qs.to(torch.int32) | ((qh & 7) << 8).reshape((n_blocks, -1))

    grid = _grid_lookup("IQ1_S", idx, d.dtype).reshape((n_blocks, -1, 4, 8))
    return (dl * (grid + delta)).reshape((n_blocks, -1))

def dequantize_blocks_IQ1_M(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    qs, qh, scales = split_block_dims(blocks, QK_K // 8, QK_K // 16)
    scales = bytes_to_u16(scales, 4)                         # (n, 4)

    # the f16 super-block scale is scattered over the top nibbles of the 4 scale words
    d = (scales & 0xF000) >> _shifts(scales.device, (12, 8, 4, 0), torch.int32).reshape((1, 4))
    d = d[:, 0] | d[:, 1] | d[:, 2] | d[:, 3]
    d = torch.where(d >= 32768, d - 65536, d).to(torch.int16)
    d = d.view(torch.float16).reshape((n_blocks, 1))
    if dtype is not None:
        d = d.to(dtype)

    sc = scales.reshape((n_blocks, -1, 1)) >> _shifts(d.device, (0, 3, 6, 9), torch.int32).reshape((1, 1, 4))
    sc = (sc & 0x07).reshape((n_blocks, -1))                 # (n, 16)
    dl = d * (2 * sc + 1).to(d.dtype)
    dl = dl.reshape((n_blocks, -1, 2, 1, 1))

    qh = qh.reshape((n_blocks, -1, 1)) >> _shifts(d.device, (0, 4)).reshape((1, 1, 2))
    qh = qh.reshape((n_blocks, -1))                          # (n, 32)
    idx = qs.to(torch.int32) | ((qh & 0x07).to(torch.int32) << 8)

    delta = torch.where((qh & 0x08) == 0, IQ1_DELTA, -IQ1_DELTA).to(d.dtype)
    delta = delta.reshape((n_blocks, -1, 2, 2, 1))

    grid = _grid_lookup("IQ1_S", idx, d.dtype).reshape((n_blocks, -1, 2, 2, 8))
    return (dl * (grid + delta)).reshape((n_blocks, -1))

# Microscaling FP4 #
def _e8m0_to_fp32_half(e):
    e = e.to(torch.int32)
    bits = torch.where(e < 2, 0x00200000 << e, (e - 1) << 23)
    return bits.view(torch.float32)

def dequantize_blocks_MXFP4(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    e, qs = split_block_dims(blocks, 1)
    d = _e8m0_to_fp32_half(e)                                # (n, 1) float32
    if dtype is not None:
        d = d.to(dtype)

    qs = qs.reshape((n_blocks, 1, block_size // 2)) >> _shifts(d.device, (0, 4)).reshape((1, 2, 1))
    qs = (qs & 0x0F).reshape((n_blocks, block_size)).long()
    return d * _kvalues_fp4(d.device)[qs]

def _ue4m3_to_fp32_half(x):
    x = x.to(torch.int32)
    exp = (x >> 3) & 0xF
    man = (x & 0x7).to(torch.float32)
    raw = torch.where(
        exp == 0,
        man * (2.0 ** -9),
        (1.0 + man / 8.0) * torch.exp2((exp - 7).to(torch.float32)),
    )
    return torch.where((x == 0) | (x == 0x7F), torch.zeros_like(raw), raw * 0.5)

def dequantize_blocks_NVFP4(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, qs = split_block_dims(blocks, 4)
    d = _ue4m3_to_fp32_half(d).reshape((n_blocks, 4, 1))
    if dtype is not None:
        d = d.to(dtype)

    qs = qs.reshape((n_blocks, 4, 8))
    vals = torch.cat([qs & 0x0F, qs >> 4], dim=-1).long()    # (n, 4, 16)
    return (d * _kvalues_fp4(d.device)[vals]).reshape((n_blocks, block_size))

_dequant_table = {
    "BF16": dequantize_blocks_BF16,
    "Q8_0": dequantize_blocks_Q8_0,
    "Q5_1": dequantize_blocks_Q5_1,
    "Q5_0": dequantize_blocks_Q5_0,
    "Q4_1": dequantize_blocks_Q4_1,
    "Q4_0": dequantize_blocks_Q4_0,
    "Q1_0": dequantize_blocks_Q1_0,
    "Q2_0": dequantize_blocks_Q2_0,
    "Q8_K": dequantize_blocks_Q8_K,
    "Q6_K": dequantize_blocks_Q6_K,
    "Q5_K": dequantize_blocks_Q5_K,
    "Q4_K": dequantize_blocks_Q4_K,
    "Q3_K": dequantize_blocks_Q3_K,
    "Q2_K": dequantize_blocks_Q2_K,
    "TQ1_0": dequantize_blocks_TQ1_0,
    "TQ2_0": dequantize_blocks_TQ2_0,
    "IQ4_NL": dequantize_blocks_IQ4_NL,
    "IQ4_XS": dequantize_blocks_IQ4_XS,
    "IQ3_S": dequantize_blocks_IQ3_S,
    "IQ3_XXS": dequantize_blocks_IQ3_XXS,
    "IQ2_S": dequantize_blocks_IQ2_S,
    "IQ2_XS": dequantize_blocks_IQ2_XS,
    "IQ2_XXS": dequantize_blocks_IQ2_XXS,
    "IQ1_M": dequantize_blocks_IQ1_M,
    "IQ1_S": dequantize_blocks_IQ1_S,
    "MXFP4": dequantize_blocks_MXFP4,
    "NVFP4": dequantize_blocks_NVFP4,
}

if hasattr(torch, "float8_e4m3fn"):
    _dequant_table["F8_E4M3"] = dequantize_blocks_F8_E4M3

# only register types known to the installed gguf-py (sizes come from GGML_QUANT_SIZES)
dequantize_functions = {
    _qt(name): fn for name, fn in _dequant_table.items()
    if _qt(name) is not None and _qt(name) in gguf.GGML_QUANT_SIZES
}

def supported_qtypes():
    return sorted(q.name for q in dequantize_functions)
