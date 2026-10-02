"""Started by the container CMD: loads the model once, then answers app.py over localhost TCP."""
import json, os, socketserver, sys, threading, time, traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
HOST = "127.0.0.1"
PORT = int(os.environ.get("OCR_PORT", "8765"))

READY = threading.Event()
LOCK = threading.Lock()
ENGINE = None


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        try:
            req = json.loads(self.rfile.readline().decode() or "{}")
            if req.get("cmd") == "ping":
                resp = {"ok": True, "ready": READY.is_set()}
            else:
                if not READY.wait(timeout=float(req.get("wait", 25))):
                    resp = {"ok": False, "error": "model not ready"}
                else:
                    with LOCK:
                        resp = {"ok": True, **ENGINE.predict(req["image"])}
        except Exception as e:
            traceback.print_exc()
            resp = {"ok": False, "error": repr(e)}
        self.wfile.write((json.dumps(resp, ensure_ascii=False) + "\n").encode())


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    global ENGINE
    try:
        srv = Server((HOST, PORT), Handler)
    except OSError:
        print("[server] port busy - another server is already running", flush=True)
        return
    threading.Thread(target=srv.serve_forever, daemon=True).start()   # accept pings while loading
    t0 = time.time()
    print("[server] loading model ...", flush=True)
    from ocr_core import OCREngine
    for attempt in range(1, 6):                    # retry on flaky network during the weight download
        try:
            ENGINE = OCREngine()
            break
        except Exception:
            traceback.print_exc()
            print(f"[server] load attempt {attempt} failed, retrying ...", flush=True)
            time.sleep(5)
    else:
        sys.exit(1)
    ENGINE.warmup()
    READY.set()
    print(f"[server] ready in {time.time() - t0:.1f}s on {HOST}:{PORT}", flush=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
