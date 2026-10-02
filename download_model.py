"""Download the weights to OCR_MODEL_DIR (the server does the same automatically at container start)."""
from ocr_core import MODEL_ID, resolve_model_path

print("downloaded", MODEL_ID, "->", resolve_model_path())
