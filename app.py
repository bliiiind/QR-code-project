from __future__ import annotations

from io import BytesIO
import sys
import threading
import time

from flask import Flask, jsonify, render_template, request, send_file

from qr_transfer import DEFAULT_CHUNK_SIZE, frame_to_dict, make_qr_png, split_text


app = Flask(__name__)
MAX_CHUNK_SIZE = 2500


class MessageStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._version = 0
        self._message: dict[str, object] | None = None

    def set_text(self, text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, source: str = "terminal") -> dict[str, object]:
        started_at = time.perf_counter()
        frames = split_text(text, chunk_size=chunk_size)
        serialized = [frame_to_dict(frame, include_png=True) for frame in frames]
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        message = {
            "version": 0,
            "source": source,
            "text": text,
            "created_at": time.time(),
            "encode_ms": round(elapsed_ms, 3),
            "byte_length": len(text.encode("utf-8")),
            "char_length": len(text),
            "frame_count": len(frames),
            "chunk_size": chunk_size,
            "frames": serialized,
        }

        with self._lock:
            self._version += 1
            message["version"] = self._version
            self._message = message
            return dict(message)

    def current(self) -> dict[str, object]:
        with self._lock:
            if self._message is None:
                return {"version": 0, "frames": [], "frame_count": 0}
            return dict(self._message)


message_store = MessageStore()


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/current")
def api_current():
    return jsonify(message_store.current())


@app.post("/api/message")
def api_message():
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", ""))
    try:
        chunk_size = int(data.get("chunk_size", DEFAULT_CHUNK_SIZE))
    except (TypeError, ValueError):
        return jsonify({"error": "chunk_size must be an integer"}), 400
    if not 64 <= chunk_size <= MAX_CHUNK_SIZE:
        return jsonify({"error": f"chunk_size must be between 64 and {MAX_CHUNK_SIZE}"}), 400
    if not text:
        return jsonify({"error": "text cannot be empty"}), 400

    try:
        message = message_store.set_text(text, chunk_size=chunk_size, source="web")
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(message)


@app.post("/api/frames")
def api_frames():
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", ""))
    try:
        chunk_size = int(data.get("chunk_size", DEFAULT_CHUNK_SIZE))
    except (TypeError, ValueError):
        return jsonify({"error": "chunk_size must be an integer"}), 400
    if not 64 <= chunk_size <= MAX_CHUNK_SIZE:
        return jsonify({"error": f"chunk_size must be between 64 and {MAX_CHUNK_SIZE}"}), 400

    include_png = bool(data.get("include_png", False))
    try:
        frames = split_text(text, chunk_size=chunk_size)
        serialized = [frame_to_dict(frame, include_png=include_png) for frame in frames]
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify(
        {
            "byte_length": len(text.encode("utf-8")),
            "char_length": len(text),
            "frame_count": len(frames),
            "frames": serialized,
        }
    )


@app.post("/api/qr")
def api_qr():
    data = request.get_json(silent=True) or {}
    payload = str(data.get("payload", ""))
    try:
        png = make_qr_png(payload)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400
    return send_file(
        BytesIO(png),
        mimetype="image/png",
        download_name="frame.png",
    )


def terminal_input_loop(chunk_size: int = DEFAULT_CHUNK_SIZE) -> None:
    print("QR server is ready.")
    print("Type text in this terminal and press Enter to convert it into QR frames.")
    print("Open http://127.0.0.1:5000 to display the generated QR sequence.")
    while True:
        try:
            text = input("server input> ")
        except EOFError:
            return
        if not text:
            continue
        try:
            message = message_store.set_text(text, chunk_size=chunk_size)
        except Exception as exc:
            print(f"encode error: {exc}", file=sys.stderr)
            continue
        print(
            "encoded "
            f"{message['char_length']} chars / {message['byte_length']} bytes "
            f"into {message['frame_count']} QR frames "
            f"in {message['encode_ms']} ms",
            flush=True,
        )


if __name__ == "__main__":
    threading.Thread(target=terminal_input_loop, daemon=True).start()
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
