# ComfyUI-GGUF-Next || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
"""
Loader tests on tiny synthetic GGUF files (need a ComfyUI checkout, CPU is fine):

    COMFYUI_PATH=/path/to/ComfyUI python -m pytest tests/test_loader.py
"""
import importlib

import numpy as np
import pytest
import torch
import gguf

Q = gguf.GGMLQuantizationType


@pytest.fixture(scope="module")
def loader(_comfy):
    return importlib.import_module("gguf_next.loader")


@pytest.fixture(scope="module")
def convert(_comfy):
    return importlib.import_module("gguf_next.tools.convert")


def write_gguf(path, arch, tensors, kv=None, quant=None):
    """tensors: {name: np.float32 array}; quant: {name: qtype} for quantized ones."""
    w = gguf.GGUFWriter(str(path), arch)
    for k, v in (kv or {}).items():
        if isinstance(v, str):
            w.add_string(k, v)
        elif isinstance(v, int):
            w.add_uint32(k, v)
        else:
            w.add_float32(k, v)
    for name, arr in tensors.items():
        qt = (quant or {}).get(name)
        if qt is None:
            w.add_tensor(name, np.ascontiguousarray(arr, dtype=np.float32) if arr.dtype.kind == "f" else arr)
        else:
            w.add_tensor(name, gguf.quants.quantize(arr.astype(np.float32), qt), raw_dtype=qt)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return str(path)


def lcpp_permute(w, n_head):
    # LlamaModel.permute from llama.cpp convert_hf_to_gguf.py
    return w.reshape(n_head, 2, w.shape[0] // n_head // 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)


def deq(t, dq):
    # "target": compute in fp32 like the gguf-py reference (None would use fp16 scale math, as upstream)
    return dq.dequantize_tensor(t, dtype=torch.float32, dequant_dtype="target") if dq.is_quantized(t) else t.float()


def tiny_llama(rng, n_head, n_kv, hidden=64):
    head_dim = hidden // n_head
    return {
        "q": rng.standard_normal((hidden, hidden)).astype(np.float32),
        "k": rng.standard_normal((n_kv * head_dim, hidden)).astype(np.float32),
        "v": rng.standard_normal((n_kv * head_dim, hidden)).astype(np.float32),
        "emb": rng.standard_normal((100, hidden)).astype(np.float32),
        "norm": rng.standard_normal(hidden).astype(np.float32),
    }


@pytest.mark.parametrize("arch,permuted", [("llama", True), ("mistral3", True), ("qwen2", False), ("qwen3", False)])
def test_llama_like_keys_and_qk_permute(loader, dq, tmp_path, arch, permuted):
    rng = np.random.default_rng(0)
    n_head, n_kv = 4, 2  # not ComfyUI's default (32, 8): the counts must come from the metadata
    m = tiny_llama(rng, n_head, n_kv)
    q = lcpp_permute(m["q"], n_head) if permuted else m["q"]
    k = lcpp_permute(m["k"], n_kv) if permuted else m["k"]
    path = write_gguf(tmp_path / f"{arch}.gguf", arch, {
        "token_embd.weight": m["emb"],
        "blk.0.attn_q.weight": q,
        "blk.0.attn_k.weight": k,
        "blk.0.attn_v.weight": m["v"],
        "blk.0.attn_norm.weight": m["norm"],
        "output_norm.weight": m["norm"],
    }, kv={f"{arch}.attention.head_count": n_head, f"{arch}.attention.head_count_kv": n_kv},
       quant={"blk.0.attn_q.weight": Q.Q8_0, "blk.0.attn_k.weight": Q.Q8_0})
    sd = loader.gguf_clip_loader(path)
    for key in ("model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight", "model.layers.0.self_attn.k_proj.weight",
                "model.layers.0.self_attn.v_proj.weight", "model.layers.0.input_layernorm.weight", "model.norm.weight"):
        assert key in sd, (arch, key, sorted(sd))
    # rows are independent Q8_0 blocks, so un-permuting the quantized rows is exact
    for name, orig in (("q_proj", m["q"]), ("k_proj", m["k"])):
        got = deq(sd[f"model.layers.0.self_attn.{name}.weight"], dq).numpy()
        ref = gguf.quants.dequantize(gguf.quants.quantize(orig, Q.Q8_0), Q.Q8_0)
        np.testing.assert_array_equal(got, ref, err_msg=f"{arch} {name}")
    np.testing.assert_array_equal(deq(sd["model.layers.0.self_attn.v_proj.weight"], dq).numpy(), m["v"])


def test_gemma2_norms_and_keys(loader, dq, tmp_path):
    rng = np.random.default_rng(1)
    norm = rng.standard_normal(64).astype(np.float32)
    path = write_gguf(tmp_path / "gemma2.gguf", "gemma2", {
        "token_embd.weight": rng.standard_normal((100, 64)).astype(np.float32),
        "blk.0.attn_norm.weight": norm + 1.0,         # llama.cpp stores Gemma norms as w + 1
        "blk.0.post_attention_norm.weight": norm + 1.0,
        "blk.0.ffn_norm.weight": norm + 1.0,
        "blk.0.post_ffw_norm.weight": norm + 1.0,
        "output_norm.weight": norm + 1.0,
        "blk.0.attn_q.weight": rng.standard_normal((64, 64)).astype(np.float32),
    })
    sd = loader.gguf_clip_loader(path)
    for key in ("model.layers.0.input_layernorm.weight", "model.layers.0.post_attention_layernorm.weight",
                "model.layers.0.pre_feedforward_layernorm.weight", "model.layers.0.post_feedforward_layernorm.weight",
                "model.norm.weight"):
        np.testing.assert_allclose(deq(sd[key], dq).numpy(), norm, atol=1e-6, err_msg=key)
    assert "model.layers.0.self_attn.q_proj.weight" in sd


def test_comfy_layout_text_passthrough(loader, dq, tmp_path):
    # e.g. a quantized CLIP-L or a Gemma 4 conversion with ComfyUI tensor names
    rng = np.random.default_rng(2)
    w = rng.standard_normal((64, 64)).astype(np.float32)
    names = {"text_model.encoder.layers.0.mlp.fc1.weight": w, "text_model.final_layer_norm.weight": w[0]}
    for arch in ("gemma4", "some_new_llm"):
        sd = loader.gguf_clip_loader(write_gguf(tmp_path / f"{arch}.gguf", arch, names, quant={"text_model.encoder.layers.0.mlp.fc1.weight": Q.Q8_0}))
        assert set(sd) == set(names), arch
    # llama.cpp names under an arch we can't map must be refused, not loaded garbled
    with pytest.raises(ValueError):
        loader.gguf_clip_loader(write_gguf(tmp_path / "unk.gguf", "some_new_llm", {"blk.0.attn_q.weight": w}))


def test_unet_loader_arch_handling(loader, tmp_path):
    rng = np.random.default_rng(3)
    w = rng.standard_normal((64, 64)).astype(np.float32)
    # unknown diffusion arch: accepted, ComfyUI's detection decides
    sd, extra = loader.gguf_sd_loader(write_gguf(tmp_path / "new.gguf", "brand_new_dit", {"blocks.0.attn.qkv.weight": w}))
    assert extra["arch_str"] == "brand_new_dit" and "blocks.0.attn.qkv.weight" in sd
    # a text encoder picked in the UNet loader gets a clear message
    with pytest.raises(ValueError, match="CLIP loader"):
        loader.gguf_sd_loader(write_gguf(tmp_path / "te.gguf", "qwen3", {"blk.0.attn_q.weight": w}))


def test_tensor_kinds(loader, dq, tmp_path):
    rng = np.random.default_rng(4)
    table = rng.standard_normal((4, 256)).astype(np.float32)
    vec = rng.standard_normal(256).astype(np.float32)
    ids = np.arange(10, dtype=np.int32)
    path = write_gguf(tmp_path / "kinds.gguf", "flux", {
        "double_blocks.0.img_attn.proj.weight": rng.standard_normal((64, 256)).astype(np.float32),
        "scale_shift_table": table,
        "final_layer.norm.weight": vec,
        "pos_ids": ids,
    }, quant={"double_blocks.0.img_attn.proj.weight": Q.Q8_0, "scale_shift_table": Q.Q8_0, "final_layer.norm.weight": Q.Q8_0})
    sd, extra = loader.gguf_sd_loader(path)
    assert dq.is_quantized(sd["double_blocks.0.img_attn.proj.weight"])
    assert sd["double_blocks.0.img_attn.proj.weight"].shape == (64, 256)
    # quantized 1D tensors are dequantized on load
    assert not dq.is_quantized(sd["final_layer.norm.weight"]) and sd["final_layer.norm.weight"].dtype == torch.float32
    # quantized bare 2D parameter keeps its shape (dequantized later by dequant_on_copy)
    assert sd["scale_shift_table"].shape == (4, 256)
    # integer payloads stay plain integer tensors
    assert sd["pos_ids"].dtype == torch.int32 and torch.equal(sd["pos_ids"], torch.from_numpy(ids))
    assert extra["qtypes"]["Q8_0"] == 3


def test_strip_quant_suffix(loader):
    cases = {
        "qwen2.5-vl-7b-instruct.q4_k_m": "qwen2.5-vl-7b-instruct",
        "qwen3-vl-32b-instruct-minimax-h3-l0-49-ud-q2_k_xl": "qwen3-vl-32b-instruct-minimax-h3-l0-49",
        "model-iq4_xs": "model",
        "model_q8_0": "model",
        "model-bf16": "model",
        "model.f16": "model",
        "plainname": "plainname",
    }
    for src, dst in cases.items():
        assert loader.strip_quant_suffix(src) == dst, src


def test_find_mmproj(loader, tmp_path):
    rng = np.random.default_rng(5)
    w = {"v.blk.0.ln1.weight": rng.standard_normal(64).astype(np.float32)}

    def mmproj(path, projector):
        return write_gguf(path, "clip", w, kv={"general.type": "mmproj", "clip.projector_type": projector})

    exp = loader.MMPROJ_PROJECTORS["qwen2vl"]
    # matched by name, quant suffix stripped
    d = tmp_path / "named"
    d.mkdir()
    te = write_gguf(d / "Qwen2.5-VL-7B.Q4_K_M.gguf", "qwen2vl", w)
    mm = mmproj(d / "Qwen2.5-VL-7B.mmproj-f16.gguf", "qwen2.5vl_merger")
    mmproj(d / "Other-Model-mmproj.gguf", "qwen3vl_merger")
    assert loader.find_mmproj(te, exp)[0] == mm
    # renamed files: the single mmproj is used only if its projector fits
    d = tmp_path / "single_ok"
    d.mkdir()
    te = write_gguf(d / "renamed-te.gguf", "qwen2vl", w)
    mm = mmproj(d / "mmproj-whatever.gguf", "qwen2vl_merger")
    assert loader.find_mmproj(te, exp)[0] == mm
    d = tmp_path / "single_wrong"
    d.mkdir()
    te = write_gguf(d / "renamed-te.gguf", "qwen2vl", w)
    mmproj(d / "mmproj-whatever.gguf", "qwen3vl_merger")
    assert loader.find_mmproj(te, exp)[0] is None


def test_convert_fingerprints(convert):
    # every fingerprint must detect its own arch: none is shadowed by an earlier, looser one
    for cls in convert.arch_list:
        for keys in cls.keys_detect:
            sd = set(keys)
            if any(k in cls.keys_banned for k in keys):
                with pytest.raises(AssertionError):
                    convert.detect_arch(sd)
                continue
            got = convert.detect_arch(sd)
            assert type(got) is cls, (cls.__name__, keys, type(got).__name__)
    with pytest.raises(AssertionError):
        convert.detect_arch({"something.unknown.weight"})
    assert convert.detect_arch({"something.unknown.weight"}, fallback="comfy_dit").arch == "comfy_dit"


def test_release_mmap_policy(_comfy, monkeypatch):
    nodes = importlib.import_module("gguf_next.nodes")
    pol = nodes.release_mmap_policy
    monkeypatch.delenv("GGUF_RELEASE_MMAP", raising=False)
    assert pol(1 << 30, available_bytes=8 << 30)  # fits: copy out of the mmap, as upstream
    assert not pol(15 << 30, available_bytes=8 << 30)  # bigger than free RAM: stay file-backed
    monkeypatch.setenv("GGUF_RELEASE_MMAP", "1")
    assert pol(15 << 30, available_bytes=8 << 30)
    monkeypatch.setenv("GGUF_RELEASE_MMAP", "0")
    assert not pol(1 << 30, available_bytes=8 << 30)
