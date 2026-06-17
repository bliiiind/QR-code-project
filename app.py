from __future__ import annotations

import argparse
from io import BytesIO
import base64
import sys
import threading
import time

from flask import Flask, jsonify, render_template, request, send_file

from qr_transfer import DEFAULT_CHUNK_SIZE, frame_to_dict, make_qr_png, split_text


app = Flask(__name__)
MAX_CHUNK_SIZE = 2500
DEFAULT_FRAME_INTERVAL_MS = 80


def _int_arg(name: str, default: int, minimum: int | None = None, maximum: int | None = None) -> int:
    raw = request.args.get(name, default)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def _png_from_payload(payload: str) -> bytes:
    box_size = _int_arg("box_size", 8, 1, 64)
    border = _int_arg("border", 4, 0, 16)
    png = make_qr_png(payload, box_size=box_size, border=border)
    size = _int_arg("size", 0, 0, 4096)
    if not size:
        return png

    from PIL import Image

    image = Image.open(BytesIO(png)).convert("RGB")
    image = image.resize((size, size), Image.Resampling.NEAREST)
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


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

    def frame(self, index: int) -> tuple[dict[str, object], dict[str, object]] | None:
        with self._lock:
            if self._message is None:
                return None
            frames = self._message.get("frames", [])
            if not isinstance(frames, list) or not frames:
                return None
            frame = frames[index % len(frames)]
            if not isinstance(frame, dict):
                return None
            return dict(self._message), dict(frame)


message_store = MessageStore()


class DisplayCursor:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._version = -1
        self._index = 0

    def next_index(self, version: int, total: int) -> int:
        with self._lock:
            total = max(1, total)
            if version != self._version:
                self._version = version
                self._index = 1 % total
                return 0
            index = self._index
            self._index = (self._index + 1) % total
            return index


display_cursor = DisplayCursor()


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


@app.get("/api/frame/<int:index>.png")
def api_frame_png(index: int):
    current = message_store.frame(index)
    if current is None:
        return jsonify({"error": "no QR message is available"}), 404
    message, frame = current
    png = _png_from_payload(str(frame["payload"]))
    response = send_file(BytesIO(png), mimetype="image/png", download_name=f"frame-{int(frame['index']) + 1}.png")
    response.headers["X-QR-Version"] = str(message["version"])
    response.headers["X-QR-Frame-Index"] = str(frame["index"])
    response.headers["X-QR-Frame-Total"] = str(frame["total"])
    response.headers["X-QR-Checksum"] = str(frame["checksum"])
    return response


@app.get("/api/display/next.png")
def api_display_next_png():
    message = message_store.current()
    frames = message.get("frames", [])
    if not isinstance(frames, list) or not frames:
        return jsonify({"error": "no QR message is available"}), 404

    version = int(message.get("version", 0))
    frame_count = int(message.get("frame_count", len(frames)))
    index = display_cursor.next_index(version, frame_count)
    frame = frames[index % len(frames)]
    if not isinstance(frame, dict):
        return jsonify({"error": "invalid QR frame"}), 500

    png = _png_from_payload(str(frame["payload"]))
    response = send_file(BytesIO(png), mimetype="image/png", download_name="microled-frame.png")
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-QR-Version"] = str(version)
    response.headers["X-QR-Frame-Index"] = str(frame["index"])
    response.headers["X-QR-Frame-Total"] = str(frame["total"])
    response.headers["X-QR-Interval-Ms"] = str(_int_arg("interval_ms", DEFAULT_FRAME_INTERVAL_MS, 20, 5000))
    response.headers["X-QR-Checksum"] = str(frame["checksum"])
    return response


@app.get("/api/display/next")
def api_display_next_json():
    message = message_store.current()
    frames = message.get("frames", [])
    if not isinstance(frames, list) or not frames:
        return jsonify({"error": "no QR message is available"}), 404

    version = int(message.get("version", 0))
    frame_count = int(message.get("frame_count", len(frames)))
    index = display_cursor.next_index(version, frame_count)
    frame = frames[index % len(frames)]
    if not isinstance(frame, dict):
        return jsonify({"error": "invalid QR frame"}), 500

    png = _png_from_payload(str(frame["payload"]))
    return jsonify(
        {
            "version": version,
            "index": frame["index"],
            "total": frame["total"],
            "checksum": frame["checksum"],
            "interval_ms": _int_arg("interval_ms", DEFAULT_FRAME_INTERVAL_MS, 20, 5000),
            "png_base64": base64.b64encode(png).decode("ascii"),
            "payload": frame["payload"],
        }
    )


def terminal_input_loop(chunk_size: int = DEFAULT_CHUNK_SIZE) -> None:
    print("QR server is ready.")
    print("Type text in this terminal and press Enter to convert it into QR frames.")
    print("MicroLED pull endpoint: /api/display/next.png")
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


def main() -> int:
    parser = argparse.ArgumentParser(description="QR text transfer server")
    parser.add_argument("--host", default="0.0.0.0", help="bind address; use 0.0.0.0 on Linux servers")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--no-terminal-input", action="store_true", help="disable stdin input loop for systemd/headless services")
    parser.add_argument("--initial-text", help="optional text to encode immediately on startup")
    args = parser.parse_args()

    if args.initial_text:
        message = message_store.set_text(args.initial_text, chunk_size=args.chunk_size, source="startup")
        print(f"initial text encoded into {message['frame_count']} QR frames", flush=True)

    if not args.no_terminal_input:
        threading.Thread(target=terminal_input_loop, args=(args.chunk_size,), daemon=True).start()

    print(f"QR server listening on http://{args.host}:{args.port}", flush=True)
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
