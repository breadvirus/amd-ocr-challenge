"""Harness entry point:  python3 /app/app.py --input-image /app/input/image_01.png
Writes /app/output/<name>_output.json  ->  {"text": "...", "confidence": 0.94}"""
import argparse, json, os, socket, subprocess, sys, time
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
HOST, PORT = "127.0.0.1", int(os.environ.get("OCR_PORT", "8765"))
T0 = time.time()
BUDGET = 26.0  # seconds; harness limit is 30 s per image - never exceed it (that zeroes the whole run)


def ask(payload, timeout):
    with socket.create_connection((HOST, PORT), timeout=timeout) as s:
        s.settimeout(timeout)
        s.sendall((json.dumps(payload) + "\n").encode())
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
    return json.loads(buf.decode())


def start_server_background():
    subprocess.Popen(
        [sys.executable, os.path.join(HERE, "server.py")],
        stdout=open("/tmp/ocr_server.log", "a"), stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, start_new_session=True,
    )


def run(image_path):
    """Ask the model server (started by the container CMD). If it isn't running, start it and keep
    waiting. If the model is still downloading/loading when the budget runs out, raise -> main()
    writes an empty answer: losing one image is far better than blowing the 30 s limit."""
    spawned = False
    while True:
        left = BUDGET - (time.time() - T0)
        if left < 1.0:
            raise TimeoutError("model not ready within the per-image budget")
        try:
            r = ask({"image": image_path, "wait": max(0.5, left - 1.0)}, timeout=left)
        except (ConnectionRefusedError, FileNotFoundError):
            if not spawned:
                start_server_background()
                spawned = True
            time.sleep(1.0)
            continue
        if r.get("ok"):
            return r
        raise RuntimeError(r.get("error", "server error"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-image", required=True)
    ap.add_argument("--output-dir", default=os.environ.get("OCR_OUTPUT_DIR", "/app/output"))
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, Path(args.input_image).stem + "_output.json")
    result = {"text": "", "confidence": 0.0}
    try:
        r = run(os.path.abspath(args.input_image))
        result = {"text": str(r.get("text", "")), "confidence": float(r.get("confidence", 0.0))}
    except Exception as e:
        print(f"[app] error: {e!r}", file=sys.stderr)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
