# Models

**English** · [Русский](../models.md)

## Diffusion models

GGUF-Next does not check `general.architecture` against a whitelist. If your ComfyUI version supports a model, its GGUF loads: ComfyUI detects the model from the tensor names. An unknown arch string gives a warning in the log, not an error.

Examples: Flux 1/2/3, Chroma, SD3.x, AuraFlow, HiDream / HiDream-O1, Cosmos / Anima, LTX-Video / LTX-2.x (AV), HunyuanVideo 1/1.5, Wan 2.x, Lumina 2 / Z-Image, Qwen-Image 1/2.1, Kandinsky 5, MiniMax H3, Krea 2, Ideogram 4, Omnigen2, Mochi, PixArt, HunyuanDiT.

GGUFs without metadata (stable-diffusion.cpp and other converters) are first matched against the key fingerprints in `tools/convert.py`. If no fingerprint matches, ComfyUI's detection decides.

Checked on real files:

| File | What was checked |
| --- | --- |
| Anima turbo Q4_K_M | full generation, bit-identical to upstream, LoRA, TorchCompileModel, run through a ComfyUI server |
| LTX-2.5 Distilled Q4_K_M | detected as `LTXAV`, full text-to-video generation with the Gemma 4 GGUF encoder |
| MiniMax H3 Q4_K (no metadata, sd.cpp) | detected as `MiniMaxH3`, full text-to-video generation with the Qwen3-VL-32B GGUF encoder. city96 and Nif00 fail with "unknown architecture" |

## Text encoders

| GGUF arch | Used by | Notes |
| --- | --- | --- |
| `t5`, `t5encoder` | Flux, SD3, Wan (UMT5) | UMT5 tokenizer rebuilt from metadata |
| `llama` | HiDream (Llama 3.1), Flux 2 (Mistral Small 3.x) | Q/K un-permute uses the head counts from the metadata |
| `mistral3` | Ernie-Image (Ministral 3 3B) | tekken tokenizer rebuilt |
| `qwen2`, `qwen2vl` | Qwen-Image, HunyuanVideo 1.5, Kandinsky 5, Omnigen2, LongCat | + `mmproj` |
| `qwen3` | Z-Image, Flux 2 klein, ACE-Step 1.5, Anima | |
| `qwen3vl` | Krea 2, FLUX.3, Qwen-Image 2.1, JoyImage, Boogu, MiniMax H3 (32B) | + `mmproj`. The 32B is detected as 32B, not as 8B |
| `gemma2` | Lumina 2 | norm `−1` fix, sentencepiece tokenizer |
| `gemma3` | NewBie, LTX-2 | |
| any, with **ComfyUI tensor names** | Gemma 4 for LTX-2.5, quantized CLIP-L/G and T5 | loaded as-is |

Checked on real files:

| File | What was checked |
| --- | --- |
| Qwen2.5-VL-7B Q4_K_M + mmproj | prompt encoding is bit-identical to upstream |
| Qwen3-VL-32B UD-Q2_K_XL + mmproj (MiniMax H3) | 902 tensors, no missing or unexpected keys, encoding, MiniMax H3 generation. city96, leejet, Nif00, molbal and ChrisColeTech fail on this file |
| Gemma 4 12B Q5_K_M (LTX-2.5) | 682 tensors, no missing keys, encoding, LTX-2.5 generation. city96, leejet, Nif00 and ChrisColeTech fail on this file; molbal gives the same encoding |

On a 4 GB card the Gemma 4 12B encoder for LTX-2.5 runs only on the CPU, with `--novram` (about 4 minutes per prompt on a laptop CPU). The limit is ComfyUI's encoder code, not the GGUF pack: it pads the prompt to 1024 tokens, keeps the output of all 49 layers and concatenates them. With no weights on the GPU at all that still needs more than 3.7 GB, and `--reserve-vram` does not help. Qwen3-VL-32B for MiniMax H3 runs on the GPU of the same card in the automatic mode, with a 2.8 GB peak.

### mmproj

Qwen2.5-VL and Qwen3-VL need an `mmproj-*.gguf` file in the same folder as the encoder.

- The loader compares file names without the quant suffix: `Model.Q4_K_M.gguf` finds `Model.mmproj-f16.gguf`.
- If no name matches and the folder holds exactly one `mmproj`, it is used only if its `clip.projector_type` fits the encoder.
- Without an `mmproj`, Qwen2.5-VL loads with a warning: text works, image input does not. Qwen3-VL does not load without one.

## Limitations

- Qwen3.5 (hybrid linear attention), GPT-OSS (MoE) and Gemma 4 in llama.cpp tensor layout are not converted to HF layout. They load only from GGUFs with ComfyUI tensor names.
- A file with llama.cpp names and an unknown arch does not load: the loader raises an error instead of returning a model with scrambled weights.
- Large token embeddings (64k rows or more) are dequantized to fp16 on load, as in upstream. Tied-embedding generation goes through ComfyUI's cast path, which can't read quantized data.
