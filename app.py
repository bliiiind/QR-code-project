from __future__ import annotations

from io import BytesIO
import logging
import sys
import threading
import time

from flask import Flask, jsonify, render_template, request, send_file

from qr_transfer import DEFAULT_CHUNK_SIZE, frame_to_dict, make_qr_png, split_text


app = Flask(__name__)
MAX_CHUNK_SIZE = 2500
TEXT_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "gbk", "big5")


def decode_text_file(data: bytes) -> tuple[str, str]:
    for encoding in TEXT_ENCODINGS:
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace"), "utf-8-replace"


class MessageStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._version = 0
        self._message: dict[str, object] | None = None

    def set_text(self, text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, source: str = "terminal") -> dict[str, object]:
        started_at = time.perf_counter()
        frames = split_text(text, chunk_size=chunk_size)
        message = {
            "version": 0,
            "source": source,
            "text": text,
            "created_at": time.time(),
            "encode_ms": None,
            "byte_length": len(text.encode("utf-8")),
            "char_length": len(text),
            "frame_count": len(frames),
            "generated_frame_count": 0,
            "chunk_size": chunk_size,
            "encoding": True,
            "frames": [],
        }

        with self._lock:
            self._version += 1
            message["version"] = self._version
            self._message = message
            version = self._version

        threading.Thread(
            target=self._generate_frames,
            args=(version, frames, started_at),
            daemon=True,
        ).start()
        return self.current()

    def _generate_frames(self, version: int, frames: list[object], started_at: float) -> None:
        for frame in frames:
            serialized = frame_to_dict(frame, include_png=True)
            with self._lock:
                if self._message is None or self._message.get("version") != version:
                    return
                stored_frames = self._message["frames"]
                if isinstance(stored_frames, list):
                    stored_frames.append(serialized)
                    self._message["generated_frame_count"] = len(stored_frames)

        elapsed_ms = (time.perf_counter() - started_at) * 1000
        with self._lock:
            if self._message is None or self._message.get("version") != version:
                return
            self._message["encode_ms"] = round(elapsed_ms, 3)
            self._message["encoding"] = False

    def current(self) -> dict[str, object]:
        with self._lock:
            if self._message is None:
                return {"version": 0, "frames": [], "frame_count": 0, "generated_frame_count": 0, "encoding": False}
            message = dict(self._message)
            frames = self._message.get("frames", [])
            message["frames"] = list(frames) if isinstance(frames, list) else []
            return message


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


@app.post("/api/message-file")
def api_message_file():
    uploaded = request.files.get("file")
    if uploaded is None:
        return jsonify({"error": "file is required"}), 400

    try:
        chunk_size = int(request.form.get("chunk_size", DEFAULT_CHUNK_SIZE))
    except (TypeError, ValueError):
        return jsonify({"error": "chunk_size must be an integer"}), 400
    if not 64 <= chunk_size <= MAX_CHUNK_SIZE:
        return jsonify({"error": f"chunk_size must be between 64 and {MAX_CHUNK_SIZE}"}), 400

    text, encoding = decode_text_file(uploaded.read())
    if not text:
        return jsonify({"error": "file is empty"}), 400

    try:
        message = message_store.set_text(text, chunk_size=chunk_size, source=f"file:{uploaded.filename or 'sample'}")
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400
    message["file_encoding"] = encoding
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
            "queued "
            f"{message['char_length']} chars / {message['byte_length']} bytes "
            f"as {message['frame_count']} QR frames",
            flush=True,
        )


if __name__ == "__main__":
    logging.getLogger("werkzeug").disabled = True
    threading.Thread(target=terminal_input_loop, daemon=True).start()
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
