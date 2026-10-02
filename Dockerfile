# Final stage MUST start from the mandated ROCm base. Never use --squash / FROM scratch.
FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

ARG OCR_MODEL_ID=Qwen/Qwen3-VL-8B-Instruct
ENV PYTHONUNBUFFERED=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_XET_HIGH_PERFORMANCE=1 \
    TOKENIZERS_PARALLELISM=false \
    OCR_MODEL_ID=${OCR_MODEL_ID} \
    OCR_MODEL_DIR=/models/ocr-vlm \
    OCR_OUTPUT_DIR=/app/output

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Weights are NOT baked in (keeps the push small). server.py downloads them from the
# Hugging Face Hub into /models at container start, inside the 10-minute startup budget.
# To bake them instead, uncomment:  RUN python3 /app/download_model.py
COPY app.py ocr_core.py server.py download_model.py /app/
RUN mkdir -p /app/input /app/output /models && chmod 777 /models

# Download (first start) + load the model once; app.py talks to it on 127.0.0.1:8765
CMD ["python3", "/app/server.py"]
