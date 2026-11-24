# Third-party notices

ComfyUI-GGUF-Next is licensed under the Apache License 2.0 (see `LICENSE` and `NOTICE`). It contains material derived from the projects below, whose licenses require the following notices to be kept.

## gguf-py and ggml / llama.cpp: MIT License

Used for:

- `dequant.py`: the torch dequantization kernels are ports of the numpy reference implementations in gguf-py (`gguf/quants.py`) and ggml (`ggml-quants.c`). The kernels include the IQ-quant sign table and the IQ4 / FP4 value tables. The IQ grid tables are not copied: they're read at runtime from the installed `gguf` package.
- `tools/lcpp.patch`: a patch against the llama.cpp source tree.

```
MIT License

Copyright (c) 2023 Georgi Gerganov
Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## ComfyUI-GGUF (city96) and leejet/ComfyUI-GGUF: Apache License 2.0

This repository is a fork of both. Their license is the same as this project's (`LICENSE`), and the attribution is in `NOTICE`.

## Runtime dependencies (not bundled)

| Package | License | Notes |
|---|---|---|
| `gguf` | MIT | required |
| `sentencepiece` | Apache-2.0 | already a ComfyUI dependency |
| `protobuf` | BSD-3-Clause | optional, only for rebuilding T5 / Gemma tokenizers |
| `torch`, `numpy`, `transformers` | BSD / BSD / Apache-2.0 | provided by ComfyUI, not installed by this package |
