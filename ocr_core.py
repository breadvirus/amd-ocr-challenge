"""Shared OCR logic: image loading, model, prompt, post-processing."""
import os, re, math, unicodedata
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageStat

Image.MAX_IMAGE_PIXELS = None  # test images can be very large
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")   # max-speed weight download at container start

MODEL_ID  = os.environ.get("OCR_MODEL_ID", "Qwen/Qwen3-VL-8B-Instruct")
MODEL_DIR = os.environ.get("OCR_MODEL_DIR", "/models/ocr-vlm")
MAX_SIDE  = int(os.environ.get("OCR_MAX_SIDE", "1280"))
MIN_SIDE  = int(os.environ.get("OCR_MIN_SIDE", "448"))
MAX_NEW_TOKENS = int(os.environ.get("OCR_MAX_NEW_TOKENS", "48"))

PROMPT = """You are a precise OCR engine for traffic-camera images. Transcribe the text in this image.
Reply with ONLY the transcription: no explanation, no labels, no quotes, no punctuation you did not see.

Rules:
1. Licence plate: output ONLY the registration number. Drop everything printed around it: state or
   province names (CALIFORNIA, NEW YORK, TEXAS ...), slogans and mottos (EXCELSIOR, THE LONE STAR STATE ...),
   websites, dealer frames and registration stickers.
2. Chinese plate: the leading Chinese province character and the letter after it ARE part of the number.
   Keep them, e.g. 京A·12345 or 沪B·88888.
3. Road sign (stop, speed limit, work zone, warning, advisory plaque): output every word and number printed
   on the sign, e.g. STOP, SPEED LIMIT 65, ROAD WORK AHEAD. If the sign shows only a number, output only
   that number - never add units such as MPH that are not printed.
4. Text on several lines: read top to bottom and join the lines with a single space.
5. Copy every character exactly as printed. The image may be blurry, noisy, dark, glared or seen at an angle -
   look carefully at look-alike characters (0/O/D/Q, 1/I/L, 2/Z, 5/S, 6/G, 8/B).
"""

US_BANNERS = [
    "ALABAMA","ALASKA","ARIZONA","ARKANSAS","CALIFORNIA","COLORADO","CONNECTICUT","DELAWARE","FLORIDA",
    "GEORGIA","HAWAII","IDAHO","ILLINOIS","INDIANA","IOWA","KANSAS","KENTUCKY","LOUISIANA","MAINE",
    "MARYLAND","MASSACHUSETTS","MICHIGAN","MINNESOTA","MISSISSIPPI","MISSOURI","MONTANA","NEBRASKA",
    "NEVADA","NEW HAMPSHIRE","NEW JERSEY","NEW MEXICO","NEW YORK","NORTH CAROLINA","NORTH DAKOTA","OHIO",
    "OKLAHOMA","OREGON","PENNSYLVANIA","RHODE ISLAND","SOUTH CAROLINA","SOUTH DAKOTA","TENNESSEE","TEXAS",
    "UTAH","VERMONT","VIRGINIA","WASHINGTON","WEST VIRGINIA","WISCONSIN","WYOMING","DISTRICT OF COLUMBIA",
    "THE LONE STAR STATE","LONE STAR STATE","EXCELSIOR","EMPIRE STATE","THE EMPIRE STATE","SUNSHINE STATE",
    "THE SUNSHINE STATE","GARDEN STATE","LAND OF LINCOLN","GRAND CANYON STATE","THE GOLDEN STATE",
    "GOLDEN STATE","DMV.CA.GOV","FAMOUS POTATOES","LIVE FREE OR DIE","FIRST IN FLIGHT","PEACH STATE",
]
_BANNER_RE = re.compile(r"(?<![A-Z0-9])(" + "|".join(sorted(map(re.escape, US_BANNERS), key=len, reverse=True)) + r")(?![A-Z0-9])")
_PREFIX_RE = re.compile(r"^\s*(the\s+)?(text|answer|transcription|output|result|plate(\s+number)?|licen[cs]e\s+plate(\s+number)?|sign|reading)\s*(is)?\s*[:：\-]\s*", re.I)


# ----------------------------------------------------------------------------- images
def load_image(path):
    im = Image.open(path)
    try:
        im.seek(0)                      # multi-page TIFF -> first page
    except Exception:
        pass
    im.load()
    try:
        im = ImageOps.exif_transpose(im)
    except Exception:
        pass
    if im.mode in ("I;16", "I;16B", "I;16L", "I", "F"):          # 16-bit / float TIFF
        a = np.asarray(im, dtype=np.float32)
        lo, hi = float(a.min()), float(a.max())
        a = (a - lo) / (hi - lo + 1e-6) * 255.0
        im = Image.fromarray(a.clip(0, 255).astype(np.uint8))
    if im.mode in ("RGBA", "LA", "P", "PA"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    return fit_size(im.convert("RGB"))


def fit_size(im):
    w, h = im.size
    s = 1.0
    if max(w, h) > MAX_SIDE:
        s = MAX_SIDE / max(w, h)
    elif min(w, h) < MIN_SIDE:
        s = min(MIN_SIDE / min(w, h), MAX_SIDE / max(w, h))
    if abs(s - 1.0) > 1e-3:
        im = im.resize((max(28, round(w * s)), max(28, round(h * s))), Image.LANCZOS)
    return im


def image_stats(im):
    g = im.convert("L")
    st = ImageStat.Stat(g)
    return st.mean[0], st.stddev[0]


def needs_enhance(im):
    mean, std = image_stats(im)
    return mean < 80 or std < 35


def enhance(im):
    """Denoise + stretch contrast + brighten shadows: helps low light, glare and sensor noise."""
    out = im.filter(ImageFilter.MedianFilter(3))
    out = ImageOps.autocontrast(out, cutoff=2)
    mean, _ = image_stats(out)
    if mean < 110:
        gamma = 0.6 if mean < 70 else 0.8
        lut = [round(255 * (i / 255) ** gamma) for i in range(256)]
        out = out.point(lut * 3)
    return out.filter(ImageFilter.UnsharpMask(radius=2, percent=120, threshold=3))


# ----------------------------------------------------------------------------- text
def postprocess(raw):
    t = raw or ""
    t = re.sub(r"<think>.*?</think>", " ", t, flags=re.S)
    t = t.replace("```", " ").strip()
    if t.startswith("{") and '"text"' in t:                       # model answered with JSON
        try:
            import json
            t = str(json.loads(t).get("text", t))
        except Exception:
            pass
    t = unicodedata.normalize("NFKC", t)                           # full-width -> ASCII, keeps CJK
    for ch in "•∙・‧⋅●":
        t = t.replace(ch, "·")
    lines = []
    for ln in t.splitlines():
        ln = _PREFIX_RE.sub("", ln).strip().strip("\"'`“”‘’")
        if ln:
            lines.append(ln)
    t = " ".join(lines)
    t = re.sub(r"\s+", " ", t).strip().rstrip(".,;:!").strip()
    t = t.upper()
    if t in {"NONE", "NO TEXT", "N/A", "UNREADABLE", "EMPTY"}:
        return ""
    # Safety net: if the model leaked a state banner/slogan next to a plate number, drop it.
    if re.search(r"\d", t):
        stripped = re.sub(r"\s+", " ", _BANNER_RE.sub(" ", t)).strip()
        if stripped and re.search(r"\d", stripped):
            t = stripped
    return t


def normalize(s):
    """Official grading normalization: uppercase, drop whitespace and - . · _"""
    s = (s or "").upper()
    s = re.sub(r"\s+", "", s)
    return re.sub(r"[-.·_]", "", s)


# ----------------------------------------------------------------------------- model
MARKER = ".download_complete"


def resolve_model_path():
    """Use local weights if fully downloaded, otherwise download them (resumes partial downloads)."""
    if os.path.isfile(os.path.join(MODEL_DIR, MARKER)):
        return MODEL_DIR
    from huggingface_hub import snapshot_download
    kw = dict(ignore_patterns=["*.bin", "*.pt", "*.pth", "*.gguf", "*.onnx", "original/*"], max_workers=16)
    try:
        path = snapshot_download(MODEL_ID, local_dir=MODEL_DIR, **kw)
    except OSError:                                                # read-only dir -> HF cache
        return snapshot_download(MODEL_ID, **kw)
    with open(os.path.join(path, MARKER), "w") as f:
        f.write(MODEL_ID)
    return path


class OCREngine:
    def __init__(self):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        path = resolve_model_path()
        self.processor = AutoProcessor.from_pretrained(path)
        self.model = AutoModelForImageTextToText.from_pretrained(
            path, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa",
        ).eval()

    def _read(self, im):
        torch = self.torch
        messages = [{"role": "user", "content": [
            {"type": "image", "image": im},
            {"type": "text", "text": PROMPT},
        ]}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                output_scores=True, return_dict_in_generate=True,
            )
        seq = out.sequences[0, inputs["input_ids"].shape[1]:]
        raw = self.processor.decode(seq, skip_special_tokens=True)
        logps = []
        for step, tok in zip(out.scores, seq):
            p = torch.softmax(step[0].float(), dim=-1)[tok].item()
            logps.append(math.log(max(p, 1e-9)))
        conf = math.exp(sum(logps) / len(logps)) if logps else 0.0
        return postprocess(raw), conf, raw

    def predict(self, path):
        im = load_image(path)
        text, conf, raw = self._read(im)
        if not text or conf < 0.80 or needs_enhance(im):
            t2, c2, r2 = self._read(enhance(im))
            if t2 and (not text or c2 > conf):
                text, conf, raw = t2, c2, r2
        return {"text": text, "confidence": round(float(conf), 4), "raw": raw}

    def warmup(self):
        im = Image.new("RGB", (448, 224), "white")
        self._read(im)
