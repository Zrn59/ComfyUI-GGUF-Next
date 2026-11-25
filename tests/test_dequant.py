# ComfyUI-GGUF-Next || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
"""
Validate the torch dequant kernels against gguf-py's numpy reference implementation.

    python -m pytest tests/test_dequant.py      (or)      python tests/test_dequant.py

Runs on CPU by default; set GGUF_TEST_DEVICE=cuda to test on GPU and
GGUF_TEST_COMPILE=1 to also exercise the torch.compile path.
"""
import os
import sys

import numpy as np
import torch
import gguf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dequant  # noqa: E402

DEVICE = os.environ.get("GGUF_TEST_DEVICE", "cpu")
N_BLOCKS = 257  # odd on purpose
SEED = 1234

def random_blocks(qtype, n_blocks, rng):
    block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
    raw = rng.integers(0, 256, size=(n_blocks, type_size), dtype=np.uint8)
    return raw, block_size, type_size

def has_numpy_reference(qtype):
    try:
        gguf.quants.dequantize(np.zeros(gguf.GGML_QUANT_SIZES[qtype][1], dtype=np.uint8), qtype)
        return True
    except NotImplementedError:
        return False

def run_torch(raw, qtype, block_size):
    n_blocks = raw.shape[0]
    data = torch.from_numpy(raw.copy()).to(DEVICE)
    out = dequant.dequantize(data, qtype, (n_blocks * block_size,), dtype=torch.float32)
    return out.float().cpu().numpy()

def check_against_reference(qtype):
    rng = np.random.default_rng(SEED + int(qtype))
    raw, block_size, _ = random_blocks(qtype, N_BLOCKS, rng)
    ref = gguf.quants.dequantize(raw, qtype).reshape(-1).astype(np.float32)
    got = run_torch(raw, qtype, block_size)
    assert got.shape == ref.shape, (qtype.name, got.shape, ref.shape)
    # exact: every value must equal the reference (NaN matches NaN)
    np.testing.assert_array_equal(got, ref, err_msg=qtype.name)

def test_all_reference_types():
    tested, skipped = [], []
    for qtype in dequant.dequantize_functions:
        if not has_numpy_reference(qtype):
            skipped.append(qtype.name)
            continue
        check_against_reference(qtype)
        tested.append(qtype.name)
    print(f"\nverified vs gguf-py reference: {', '.join(sorted(tested))}")
    print(f"no numpy reference available: {', '.join(sorted(skipped))}")

def test_q8_k():
    rng = np.random.default_rng(SEED)
    # Q8_K: d(f32) qs[256] bsums[16 x i16]
    q = gguf.GGMLQuantizationType.Q8_K
    if q in dequant.dequantize_functions:
        d = np.float32(0.0123)
        qs = rng.integers(-128, 128, size=256, dtype=np.int8)
        raw = np.concatenate([np.array([d], np.float32).view(np.uint8), qs.view(np.uint8), np.zeros(32, np.uint8)])
        got = run_torch(raw.reshape(1, -1), q, 256)
        np.testing.assert_allclose(got, d * qs.astype(np.float32), rtol=1e-6)

def test_q1_0():
    q = getattr(gguf.GGMLQuantizationType, "Q1_0", None)
    if q is None or q not in dequant.dequantize_functions:
        return
    d = np.float16(0.5)
    bits = np.random.default_rng(SEED).integers(0, 2, size=128).astype(np.uint8)
    packed = np.packbits(bits, bitorder="little")
    raw = np.concatenate([np.array([d], np.float16).view(np.uint8), packed]).reshape(1, -1)
    got = run_torch(raw, q, 128)
    np.testing.assert_allclose(got, np.where(bits == 1, 0.5, -0.5).astype(np.float32))

def test_low_precision_output():
    # dequant directly into fp16/bf16 should stay close to the fp32 result
    rng = np.random.default_rng(SEED)
    for name in ("Q4_K", "Q6_K", "IQ2_XS", "IQ3_S", "IQ4_XS", "Q8_0"):
        qtype = getattr(gguf.GGMLQuantizationType, name)
        block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
        # realistic data: quantize a gaussian with the reference quantizer if possible
        x = rng.standard_normal((64, block_size * 4)).astype(np.float32)
        try:
            raw = gguf.quants.quantize(x, qtype)
        except NotImplementedError:
            raw, _, _ = random_blocks(qtype, 256, rng)
            raw[:, :2] = np.array([0.01], np.float16).view(np.uint8)  # sane scale for IQ types (d first)
        data = torch.from_numpy(np.ascontiguousarray(raw)).to(DEVICE)
        shape = (data.numel() // type_size * block_size,)
        ref = dequant.dequantize(data, qtype, shape, dtype=torch.float32).float()
        for dt in (torch.float16, torch.bfloat16):
            out = dequant.dequantize(data, qtype, shape, dtype=dt).float()
            # random scale bytes can overflow fp16 intermediates; keep only sane blocks
            rb, ob = ref.reshape(-1, block_size), out.reshape(-1, block_size)
            keep = torch.isfinite(rb).all(-1) & (rb.abs().amax(-1) < 1e3)
            r, o = rb[keep], ob[keep]
            assert keep.float().mean() > 0.3, (name, "too few sane blocks")
            tol = 1e-2 if dt == torch.float16 else 4e-2
            assert torch.allclose(o, r, rtol=tol, atol=tol * r.abs().max().item()), (name, dt)

def test_compile_path():
    # torch.compile must reproduce the eager result for every type and output dtype:
    # bit for bit in fp32, and within rounding in fp16/bf16 (inductor keeps fp32 intermediates,
    # so for d*q - m types it is slightly *more* precise than eager)
    if os.environ.get("GGUF_TEST_COMPILE", "0") != "1":
        if "pytest" in sys.modules:
            sys.modules["pytest"].skip("set GGUF_TEST_COMPILE=1")
        return
    torch._dynamo.reset()
    rng = np.random.default_rng(SEED)
    try:
        for qtype in dequant.dequantize_functions:
            block_size, _ = gguf.GGML_QUANT_SIZES[qtype]
            for n_blocks in (N_BLOCKS, 64): # two sizes exercise the dynamic block count
                raw, _, _ = random_blocks(qtype, n_blocks, rng)
                data = torch.from_numpy(raw).to(DEVICE)
                shape = (n_blocks * block_size,)
                dequant.set_compile_dequant(False)
                exact = dequant.dequantize(data, qtype, shape, dtype=torch.float32)
                for dt in (torch.float32, None, torch.float16, torch.bfloat16):
                    dequant.set_compile_dequant(False)
                    ref = dequant.dequantize(data, qtype, shape, dtype=dt)
                    dequant.set_compile_dequant(True)
                    dequant.dequantize(data, qtype, shape, dtype=dt) # first call per device runs eager
                    out = dequant.dequantize(data, qtype, shape, dtype=dt)
                    assert out.dtype == ref.dtype, (qtype.name, dt)
                    # random fp16 scale bytes can overflow; compare finite values only
                    keep = torch.isfinite(ref) & torch.isfinite(out) & torch.isfinite(exact)
                    o, r, x = out[keep].float(), ref[keep].float(), exact[keep]
                    if dt == torch.float32 or ref.dtype == torch.float32:
                        assert torch.equal(o, r), (qtype.name, n_blocks, dt)
                    else:
                        eps = torch.finfo(ref.dtype).eps
                        bound = torch.maximum((r - x).abs(), eps * x.abs()) * 2 + 1e-7
                        bad = (o - x).abs() > bound
                        assert not bad.any(), (qtype.name, n_blocks, dt, int(bad.sum()))
    finally:
        dequant.set_compile_dequant(False)
    assert not dequant._compile_failed, dequant._compile_failed

if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
