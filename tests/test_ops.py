# ComfyUI-GGUF-Next || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
"""
Integration tests that need a ComfyUI checkout (CPU is fine):

    COMFYUI_PATH=/path/to/ComfyUI python tests/test_ops.py
"""
import os
import sys
import importlib

import pytest
import torch
import gguf

COMFYUI_PATH = os.environ.get("COMFYUI_PATH")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def setup():
    if not COMFYUI_PATH:
        return None, None
    if "gguf_next" in sys.modules:
        return importlib.import_module("gguf_next.ops"), importlib.import_module("gguf_next.dequant")
    sys.argv = [sys.argv[0], "--cpu"]
    sys.path.insert(0, COMFYUI_PATH)
    import comfy.options
    comfy.options.enable_args_parsing()
    # import this repo as a package regardless of folder name
    spec = importlib.util.spec_from_file_location("gguf_next", os.path.join(REPO, "__init__.py"), submodule_search_locations=[REPO])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules["gguf_next"] = pkg
    spec.loader.exec_module(pkg)
    return importlib.import_module("gguf_next.ops"), importlib.import_module("gguf_next.dequant")

def make_q(x, qtype, ops):
    raw = gguf.quants.quantize(x.numpy(), qtype)
    return ops.GGMLTensor(torch.from_numpy(raw), tensor_type=qtype, tensor_shape=torch.Size(x.shape))

def rand_q(shape, qtype, ops, seed=0):
    # random blocks with sane fp16 scales (gguf-py can't quantize K-quants)
    block, nbytes = gguf.GGML_QUANT_SIZES[qtype]
    rows, cols = shape
    g = torch.Generator().manual_seed(seed)
    raw = torch.randint(0, 256, (rows * cols // block, nbytes), dtype=torch.uint8, generator=g)
    scale_offsets = {gguf.GGMLQuantizationType.Q4_K: (0, 2), gguf.GGMLQuantizationType.Q6_K: (nbytes - 2,)}[qtype]
    for off in scale_offsets:
        d = (torch.rand(raw.shape[0], generator=g) * 0.01 + 0.001).to(torch.float16)
        raw[:, off:off + 2] = d.view(torch.uint8).reshape(-1, 2)
    return ops.GGMLTensor(raw.reshape(rows, -1), tensor_type=qtype, tensor_shape=torch.Size(shape))

def test_copy_into_plain_parameter(ops, dq):
    # bare nn.Parameter (e.g. scale_shift_table) receiving a quantized tensor
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.table = torch.nn.Parameter(torch.zeros(4, 256))
    x = torch.randn(4, 256)
    q = make_q(x, gguf.GGMLQuantizationType.Q8_0, ops)
    m = M()
    with ops.dequant_on_copy():
        m.load_state_dict({"table": q})
    assert "__torch_function__" not in ops.GGMLTensor.__dict__, "load hook leaked into inference"
    ref = dq.dequantize_tensor(q, dtype=torch.float32)
    assert torch.allclose(m.table.data, ref), "quantized copy_ not dequantized"
    assert (m.table.data - x).abs().max() < 0.05

def test_embedding_row_gather(ops, dq):
    # the row-gather fast path must match the full-table path bit for bit, dtype included
    class FullTable(ops.GGMLOps.Embedding):
        def _can_gather_rows(self):
            return False
    ids = torch.tensor([[0, 5, 999], [17, 17, 3]])
    x = torch.randn(1000, 256)
    for qtype in (gguf.GGMLQuantizationType.Q8_0, gguf.GGMLQuantizationType.Q4_K, gguf.GGMLQuantizationType.Q6_K):
        q = make_q(x, qtype, ops) if qtype == gguf.GGMLQuantizationType.Q8_0 else rand_q(x.shape, qtype, ops)
        for dequant_dtype in (None, "target", torch.float32):
            fast, full = ops.GGMLOps.Embedding(1000, 256), FullTable(1000, 256)
            for emb in (fast, full):
                emb.load_state_dict({"weight": q})
                emb.dequant_dtype = dequant_dtype
            assert fast._can_gather_rows() and not full._can_gather_rows()
            for out_dtype in (None, torch.float32, torch.bfloat16, torch.float16):
                a, b = fast(ids, out_dtype=out_dtype), full(ids, out_dtype=out_dtype)
                assert a.shape == (2, 3, 256)
                assert a.dtype == b.dtype and torch.equal(a, b), (qtype.name, dequant_dtype, out_dtype)
            a = fast.host_rows(ids, out_dtype=torch.float16)
            assert a.dtype == torch.float16 and torch.equal(a, full(ids, out_dtype=torch.float16))

@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU: the path is for a host table with device input")
def test_embedding_host_table(ops, dq):
    # a big plain table left on the host: only the looked-up rows move, same result as the full-table path
    class FullTable(ops.GGMLOps.Embedding):
        def _host_table(self, input):
            return False
    ids = torch.tensor([[0, 5, 70000], [17, 17, 65535]], device="cuda")
    for wdtype in (torch.float16, torch.bfloat16, torch.float32):
        w = torch.randn(70001, 64).to(wdtype)
        fast, full = ops.GGMLOps.Embedding(70001, 64, dtype=wdtype), FullTable(70001, 64, dtype=wdtype)
        for emb in (fast, full):
            emb.load_state_dict({"weight": w})
        assert fast._host_table(ids) and not fast._host_table(ids.cpu())
        for out_dtype in (None, torch.float32, torch.bfloat16, torch.float16):
            a, b = fast(ids, out_dtype=out_dtype), full(ids, out_dtype=out_dtype)
            assert a.device == b.device and a.shape == (2, 3, 64)
            assert a.dtype == b.dtype and torch.equal(a, b), (wdtype, out_dtype)

def test_linear_and_conv_layers(ops, dq):
    for qtype in (gguf.GGMLQuantizationType.Q8_0, gguf.GGMLQuantizationType.Q4_0):
        lin = ops.GGMLOps.Linear(256, 64)
        w = torch.randn(64, 256) * 0.05
        lin.load_state_dict({"weight": make_q(w, qtype, ops), "bias": torch.zeros(64)}, strict=False)
        inp = torch.randn(3, 256)
        out = lin(inp)
        ref = inp @ dq.dequantize_tensor(lin.weight, dtype=torch.float32).T
        assert torch.allclose(out, ref, atol=1e-4)

    conv = ops.GGMLOps.Conv3d(8, 32, kernel_size=(1, 2, 2))
    w = torch.randn(32, 8, 1, 2, 2) * 0.1
    qw = gguf.quants.quantize(w.reshape(-1, 32).numpy(), gguf.GGMLQuantizationType.Q8_0)
    qw = ops.GGMLTensor(torch.from_numpy(qw), tensor_type=gguf.GGMLQuantizationType.Q8_0, tensor_shape=w.shape)
    conv.load_state_dict({"weight": qw, "bias": torch.zeros(32)})
    out = conv(torch.randn(1, 8, 2, 4, 4))
    assert out.shape == (1, 32, 2, 3, 3) and torch.isfinite(out).all()

def test_torch_compile_model(ops, dq):
    # regression: GGMLTensor must stay traceable for TorchCompileModel
    torch._dynamo.reset()
    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = ops.GGMLOps.Linear(256, 512)
            self.b = ops.GGMLOps.Linear(512, 256)
        def forward(self, x):
            return self.b(torch.nn.functional.gelu(self.a(x)))
    net = Net()
    for name, (o, i) in (("a", (512, 256)), ("b", (256, 512))):
        getattr(net, name).load_state_dict({"weight": make_q(torch.randn(o, i) * 0.05, gguf.GGMLQuantizationType.Q8_0, ops)}, strict=False)
    x = torch.randn(4, 256)
    ref = net(x)
    out = torch.compile(net)(x)
    assert torch.allclose(out, ref, atol=1e-3)

if __name__ == "__main__":
    ops, dq = setup()
    if ops is None:
        print("COMFYUI_PATH not set, skipping")
        sys.exit(0)
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(ops, dq)
            print(f"ok  {name}")
