# Robust Traffic OCR with Qwen3-VL on AMD ROCm

Submission for **AMD AI Academy Challenge: Mini-Challenge 2 (Optical Character Recognition)**.

This app reads licence plates and road signs from difficult images (noise, blur, glare, low light, off-axis angles) and writes the text as JSON. It runs on AMD GPUs with ROCm.

## Results (sample set)

| Metric | Result | Limit |
|---|---|---|
| Accuracy | 10/10 | |
| Startup (model load + warmup) | 16 s | 600 s |
| Slowest image | 1.2 s | 30 s |
| All 10 images | 5.2 s | 600 s |
| Peak VRAM | 18.2 GiB | 1–48 GiB |

## How it works

- **Model:** [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) (open source, bf16), loaded with Hugging Face Transformers on the `rocm/pytorch` base image.
- **Prompt:** a strict transcription prompt that follows the challenge's rules:
  - US plates: return the number only. State banners and slogans are dropped (`CALIFORNIA 7ABC123` → `7ABC123`).
  - Chinese plates: the province character and letter are kept (`京A·12345`).
  - Signs: every printed word is kept (`SPEED LIMIT 65`), and no units are added.
  - Multi-line text is read top to bottom and joined with single spaces.
- **Pre-processing:** handles PNG, JPEG and TIFF (including 16-bit and alpha), EXIF rotation, very large and very small images.
- **Enhancement retry:** dark, low-contrast or low-confidence images get a second pass on an enhanced copy (denoise, contrast stretch, gamma, sharpen). The more confident answer is kept.
- **Post-processing:** removes labels and quotes, normalises full-width characters and interpuncts, and strips state names that leak into plate readings.
- **Speed:** `server.py` loads the model once when the container starts. Each `app.py` call is a thin client, so a single image takes 0.3–1.2 s. A hard time budget means no call exceeds the 30 s per-image limit.

## Files

| File | Purpose |
|---|---|
| `app.py` | Entry point: `python3 /app/app.py --input-image <path>` writes `/app/output/<name>_output.json` |
| `ocr_core.py` | Image loading, prompt, model inference, post-processing |
| `server.py` | Persistent model server started by the container `CMD` |
| `download_model.py` | Downloads the model weights to `/models` |
| `requirements.txt` | Python dependencies (torch comes from the ROCm base image) |
| `Dockerfile` | Container build on the mandated ROCm base image |
| `amd_ocr_challenge.ipynb` | Development notebook: evaluation, robustness tests, end-to-end timing and VRAM checks |

## Output format

```json
{"text": "7ABC123", "confidence": 0.98}
```

## Build and run

```bash
docker build -t ocr-amd .
docker run -d --name ocr --device=/dev/kfd --device=/dev/dri --group-add video ocr-amd
docker exec ocr python3 /app/app.py --input-image /app/input/image_01.png
docker exec ocr cat /app/output/image_01_output.json
```

The model weights (about 17 GB) are downloaded from Hugging Face when the container first starts.
