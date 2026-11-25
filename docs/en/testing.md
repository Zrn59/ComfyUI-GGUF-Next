# Tests

**English** · [Русский](../testing.md)

```bash
pip install pytest
python -m pytest
```

With no environment variables only the CPU kernel tests run. The rest is enabled like this:

| Variable | What it adds |
| --- | --- |
| `GGUF_TEST_DEVICE=cuda` | kernels on the GPU |
| `GGUF_TEST_COMPILE=1` | the `torch.compile` path: every type, every output dtype, two tensor sizes |
| `COMFYUI_PATH=/path/to/ComfyUI` | layer and loader tests, `torch.compile` of a whole model |

Everything at once:

```bash
GGUF_TEST_DEVICE=cuda GGUF_TEST_COMPILE=1 COMFYUI_PATH=~/ComfyUI python -m pytest
```

## What is checked

- `tests/test_dequant.py`: every kernel that the installed gguf-py has a numpy reference for, with exact equality (NaN matches NaN). Q8_K has no reference, and Q1_0 has one only in master gguf-py, so both also have their own tests. fp16/bf16 output. Compiled kernels against the regular ones.
- `tests/test_ops.py`: quantized `copy_` into a plain parameter at load. Embedding row gather matches the full table bit for bit for every `out_dtype` and `dequant_dtype`, and so does the host-side gather for a large plain table (GPU only). Linear and Conv3d. `torch.compile` of a model built from GGML layers.
- `tests/test_loader.py`: synthetic GGUFs. llama.cpp → HF names for llama, mistral3, qwen2, qwen3 and gemma2. Q/K un-permute with the head counts from the metadata. Gemma norm fix. Files with ComfyUI names. Unknown archs. Tensor kinds. `mmproj` lookup. `convert.py` fingerprints don't shadow each other.

CI (`.github/workflows/tests.yml`) runs the kernels on Python 3.10 and 3.12 with `gguf` 0.13.0 and the latest one, and the integration tests with ComfyUI master and `GGUF_TEST_COMPILE=1`. It also checks that `pip install -r requirements.txt` changes no installed package.

## Manual check for 0.1.0

Before the release, on an RTX 3050 Laptop, torch 2.11, ComfyUI 0.42.1:

- The pytest suite is green on CPU and CUDA, with and without compile, on `gguf` 0.13.0, 0.17.1, 0.19.0 and master.
- Anima Q4_K_M, full 512×512 generation in separate processes, compared with city96/ComfyUI-GGUF: bit-identical for Unet Loader (GGUF), for Advanced with `dequant_dtype=target` / `patch_dtype=float32`, and with a LoRA. CLIPLoader (GGUF) gives the same image as the stock CLIPLoader.
- `compile_dequant` on every quantized tensor of the Anima file: never further from fp32 than the regular path. The image changes in details, as with ComfyUI's TorchCompileModel on city96.
- A ComfyUI server with the pack in `custom_nodes`: the nodes register in the `bootleg` category, workflows run over the HTTP API, and a warning is printed next to the original ComfyUI-GGUF.
- Six real GGUF files loaded with city96, leejet, Nif00, molbal, ChrisColeTech and GGUF-Next (table in the README). Qwen2.5-VL-7B encoding is bit-identical across city96, leejet, molbal and GGUF-Next.
- LTX-2.5 + Gemma 4 12B (1280×704, 97 frames, two-stage with the latent upscaler) and MiniMax H3 + Qwen3-VL-32B (960×544, 73 frames): full text-to-video generation with sound, all weights from GGUF. Text encoder, transformer and VAE ran in separate processes so that one large model at a time is in RAM.
- `GGUF_RELEASE_MMAP=0` and `1` give the same Anima image, with and without a LoRA, and the same as city96. With `0` the load adds 3 MB of RAM instead of 1.3 GB.
- `pip install -r requirements.txt` in a venv with ComfyUI's requirements added only `gguf` and `protobuf`; `pip check` is clean.

## Exact reference check

The 0.1.0 runs compared the kernels with the gguf-py reference within a 1e-5 tolerance. On 2026-11-25 the check was made exact. It passes on CUDA with `gguf` 0.13.0 and 0.19.0. It has not been re-run against master gguf-py yet, which is the only place Q1_0, Q2_0 and F8_E4M3 have a reference.
