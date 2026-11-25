# Changelog

## Unreleased

- `tests/test_dequant.py` compares the kernels with the gguf-py reference exactly instead of within a 1e-5 tolerance. Passes on CUDA with gguf 0.13.0 and 0.19.0.
- README and docs say precisely what is checked: 24 types against the gguf-py 0.19 reference, 26 quantized types out of 35 storage types, and bit-identical output in the comparisons that were actually run (Anima, Qwen2.5-VL-7B).

## Next 0.1.0 (fork of city96/ComfyUI-GGUF @ 6ea2651)

### Dequantization (`dequant.py`)
- Native torch kernels for IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, TQ1_0, TQ2_0, MXFP4, NVFP4, Q1_0, Q2_0, F8_E4M3 and Q8_K. These used to fall back to numpy on the CPU or weren't supported at all. Types the installed gguf-py doesn't know are skipped.
- All kernels are checked bit for bit against the gguf-py reference (`tests/test_dequant.py`). The tests run against gguf 0.13.0, 0.17.1, 0.19.0 and llama.cpp master.
- Shift and lookup tensors are cached per device, which removes a blocking host→device copy from every dequant call.
- IQ4_NL / IQ4_XS look values up by direct indexing instead of `expand` + `gather`.
- Optional `torch.compile` of the block dequant functions (`compile_dequant` on the advanced loader, or `GGUF_COMPILE_DEQUANT=1`). The first call per device runs eagerly to build the lookup tables, and a type whose compile fails falls back to eager for good. Only the block count is dynamic: with `dynamic=True`, inductor in torch 2.11 miscompiles Q2_K, Q3_K, Q6_K, TQ1_0 and TQ2_0. fp32 output is bit-identical to eager; fp16/bf16 output can differ in the last bit.
- Integer tensor types (I8/I16/I32/I64) and F64 are handled as native torch dtypes.
- Removed the unused Q8_1 type: gguf-py lists an outdated block size for it, and it never appears in model files.

### Layers (`ops.py`)
- New GGML layer types: Conv1d, Conv3d (including `autopad`), ConvTranspose1d/2d and RMSNorm.
- While a model loads (the `dequant_on_copy()` context), a quantized source in `copy_` into a regular parameter or buffer is dequantized instead of having its raw blocks copied. The hook is removed after loading, so it adds no inference overhead and doesn't affect `torch.compile`.
- A quantized `Embedding` gathers only the looked-up rows from the quantized data, and `host_rows` is supported. The result is bit-identical to the full-table path for every `out_dtype` / `dequant_dtype`, dtype included.
- A large plain `Embedding` table (64k rows or more, e.g. a token embedding kept in fp16) that stays on the host copies only the looked-up rows to the GPU, instead of the whole table on every call. For the Gemma 4 encoder of LTX-2.5 that is 1.9 GB less to move per prompt. Bit-identical to the full-table path.
- Uses `comfy.model_management.cast_to`, because `comfy.ops.cast_to` is deprecated upstream.

- `torch.compile` of a whole GGUF model (TorchCompileModel) keeps working. There's a regression test for it.

### Nodes (`nodes.py`)
- `compile_dequant` option on the advanced UNet loader.
- A startup warning when another GGUF node pack with the same node IDs is installed.
- A model bigger than free RAM is no longer copied out of the file mmap on first load (upstream does this for every model). The copy could only go to swap: LTX-2.5 Q4_K_M (15.7 GB) on 15 GB RAM takes 114–129 s per step instead of 547–580 s, with the same 2.3 GB in VRAM. `GGUF_RELEASE_MMAP=1` / `0` forces either behaviour. Output is unchanged.

### Loading (`loader.py`)
- Any diffusion `general.architecture` is accepted, with a warning when it's unknown, and ComfyUI's model detection decides. Text encoder GGUFs loaded in the UNet loader now get a clear error message.
- sd.cpp files that fail the architecture fingerprint go to ComfyUI's detection instead of raising.
- Text encoders:
  - New: `gemma2`, `qwen2`, `mistral3`.
  - Qwen3-VL `mmproj` mapping, with a key layout that depends on the variant (4B/8B HF-style vs. 32B flat).
  - Files with ComfyUI tensor names (e.g. Gemma 4 for LTX-2.5) and raw quantized CLIP/T5 dumps without an arch are passed through.
- Q/K un-permute takes `head_count` / `head_count_kv` from the metadata.
- The tekken tokenizer is rebuilt for any 131072-vocab Mistral, which includes Ministral 3 3B.
- The Gemma sentencepiece rebuild is shared between Gemma 2 and Gemma 3.
- `mmproj` lookup: `.Q4_K_M`-style suffixes are stripped. If the names don't match but the folder holds only one `mmproj` with the right projector type, that one is used.
- All quantized 1D tensors are dequantized on load, not just BF16 ones.
- Metadata extraction handles every scalar GGUF type.
- `extra` now also returns `compat` and `qtypes`.

### Tools
- `convert.py` gains about 20 architecture fingerprints (Flux 2/3, Chroma, HiDream-O1, Qwen-Image 1/2.1, LTX-AV, MiniMax H3, Krea 2, Ideogram 4, Mochi, PixArt, HunyuanDiT, Kandinsky 5, Omnigen2, Boogu, MOSS-Image, ACE-Step, Ernie-Image).
- New `--arch` override for models it doesn't recognize.
- Bare non-`.weight`/`.bias` tensors are always written in F32.

- `lcpp.patch` is regenerated against llama.cpp `b3962` and adds 19 image archs, including a generic `comfy_dit`. They share one rule set that keeps embedders, I/O projections and modulation tables in high precision. The patch builds cleanly, and an end-to-end convert → Q4_K_M → load run was tested.
- LTX-2 (AV) checkpoints are written as `ltxv`, so the existing LTXV quantization rules apply to them.

### Packaging, license, tests
- `NOTICE` and `THIRD_PARTY_NOTICES.md` added (Apache-2.0 attribution for upstream and leejet, MIT notice for the gguf-py / ggml-derived kernels and the llama.cpp patch). Modified files say so in their header.
- `requirements.txt` stays minimal and unpinned, with lower bounds only, so it never upgrades or downgrades packages in an existing ComfyUI environment.
- pytest support (`conftest.py`, `pytest.ini`) and a GitHub Actions workflow: kernel tests on Python 3.10 and 3.12 with gguf 0.13.0 and the latest gguf, plus integration tests against ComfyUI master with the compile path on. CI also checks that `requirements.txt` changes no installed package.
- `tests/test_loader.py`: synthetic GGUFs for the llama.cpp → HF key maps, the Q/K un-permute, Gemma norms, ComfyUI-layout passthrough, unknown archs, tensor kinds, `mmproj` lookup and `convert.py` fingerprints.
- README rewritten, with a Russian translation (`README.ru.md`) and paired docs pages in `docs/` and `docs/en/`.
- README shows the checks on real files: Anima generated with city96 and GGUF-Next (bit-identical), six GGUF files loaded with city96, leejet, Nif00, molbal, ChrisColeTech and GGUF-Next, and LTX-2.5 / MiniMax H3 videos made from GGUF files only. Images and videos are in `docs/`.

### Fixes carried over from leejet/ComfyUI-GGUF
- Qwen3-VL mmproj mapping and the Ideogram4 / Krea2 / MiniMax / Qwen-Image 2.1 fingerprints.
