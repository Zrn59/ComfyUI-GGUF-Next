# Тесты

[English](en/testing.md) · **Русский**

```bash
pip install pytest
python -m pytest
```

Без переменных окружения запускаются только тесты ядер на CPU. Остальное включается так:

| Переменная | Что добавляет |
| --- | --- |
| `GGUF_TEST_DEVICE=cuda` | ядра на GPU |
| `GGUF_TEST_COMPILE=1` | путь `torch.compile`: все типы, все dtype выхода, два размера тензора |
| `COMFYUI_PATH=/путь/к/ComfyUI` | тесты слоёв и загрузчика, `torch.compile` целой модели |

Всё сразу:

```bash
GGUF_TEST_DEVICE=cuda GGUF_TEST_COMPILE=1 COMFYUI_PATH=~/ComfyUI python -m pytest
```

## Что проверяется

- `tests/test_dequant.py` — каждое ядро против эталона gguf-py бит-в-бит. Отдельно Q8_K и Q1_0, для которых в gguf-py нет numpy-версии. Выход в fp16/bf16. Скомпилированные ядра против обычных.
- `tests/test_ops.py` — квантованный `copy_` в обычный параметр при загрузке. Выборка строк эмбеддинга совпадает с полной таблицей бит-в-бит для всех `out_dtype` и `dequant_dtype`, как и выборка на хосте для большой неквантованной таблицы (только на GPU). Linear и Conv3d. `torch.compile` модели из GGML-слоёв.
- `tests/test_loader.py` — синтетические GGUF: имена llama.cpp → HF для llama, mistral3, qwen2, qwen3 и gemma2. Обратная перестановка Q/K по числу голов из метаданных. Поправка норм Gemma. Файлы с именами ComfyUI. Неизвестные архитектуры. Типы тензоров. Поиск `mmproj`. Отпечатки `convert.py` не перекрывают друг друга.

CI (`.github/workflows/tests.yml`) гоняет ядра на Python 3.10 и 3.12 с `gguf` 0.13.0 и последним, а интеграционные тесты — с ComfyUI master и `GGUF_TEST_COMPILE=1`. Там же проверяется, что `pip install -r requirements.txt` не меняет уже установленные пакеты.

## Ручная проверка 0.1.0

Перед выпуском, на RTX 3050 Laptop, torch 2.11, ComfyUI 0.42.1:

- Набор pytest зелёный на CPU и CUDA, с компиляцией и без, на `gguf` 0.13.0, 0.17.1, 0.19.0 и master.
- Anima Q4_K_M, полная генерация 512×512 в отдельных процессах, сравнение с city96/ComfyUI-GGUF: совпадение бит-в-бит для Unet Loader (GGUF), для Advanced с `dequant_dtype=target` / `patch_dtype=float32`, с LoRA. CLIPLoader (GGUF) даёт ту же картинку, что и штатный CLIPLoader.
- `compile_dequant` на каждом квантованном тензоре файла Anima: ни разу не дальше от fp32, чем обычный путь. Картинка меняется в деталях, как и с TorchCompileModel ComfyUI на city96.
- Сервер ComfyUI с паком в `custom_nodes`: узлы зарегистрированы в категории `bootleg`, воркфлоу через HTTP API отрабатывают, рядом с оригинальным ComfyUI-GGUF выводится предупреждение.
- Шесть реальных GGUF-файлов загружены паками city96, leejet, Nif00 и GGUF-Next (таблица в README). Кодирование Qwen2.5-VL-7B совпадает бит-в-бит у city96, leejet и GGUF-Next.
- LTX-2.5 + Gemma 4 12B (1280×704, 97 кадров, две стадии с латентным апскейлером) и MiniMax H3 + Qwen3-VL-32B (960×544, 73 кадра): полная генерация видео со звуком по тексту, все веса из GGUF. Энкодер, трансформер и VAE запускались отдельными процессами, чтобы в RAM была одна большая модель за раз.
- `GGUF_RELEASE_MMAP=0` и `1` дают одну и ту же картинку Anima, с LoRA и без, и она совпадает с city96. С `0` загрузка добавляет 3 МБ RAM вместо 1.3 ГБ.
- `pip install -r requirements.txt` в venv с требованиями ComfyUI добавил только `gguf` и `protobuf`, `pip check` чистый.
