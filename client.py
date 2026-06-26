from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib import request as urlrequest

import numpy as np
from PIL import Image, ImageGrab

from qr_transfer import DEFAULT_CHUNK_SIZE, Frame, ProtocolError, assemble_frames, make_qr_png, parse_payload, split_text


USE_GPU_ACCELERATION = False
DEFAULT_CAPTURE_INTERVAL = 1 / 60
DEFAULT_PLAYBACK_INTERVAL = 1 / 10
TEXT_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "gbk", "big5")


def completed_message_ids(frames: dict[tuple[str, int], Frame]) -> set[str]:
    by_id: dict[str, set[int]] = {}
    totals: dict[str, int] = {}
    for frame in frames.values():
        by_id.setdefault(frame.message_id, set()).add(frame.index)
        totals[frame.message_id] = frame.total
    return {
        message_id
        for message_id, indexes in by_id.items()
        if len(indexes) >= totals.get(message_id, 0)
    }


@dataclass
class CaptureStats:
    started_at: float
    first_frame_at: float | None = None
    completed_at: float | None = None
    captured_frames: int = 0
    message_id: str | None = None

    @classmethod
    def start(cls) -> "CaptureStats":
        return cls(started_at=time.perf_counter())

    def mark_frame(self, frame: Frame) -> None:
        now = time.perf_counter()
        if self.first_frame_at is None:
            self.first_frame_at = now
        self.captured_frames += 1
        self.message_id = frame.message_id

    def mark_complete(self) -> None:
        if self.completed_at is None:
            self.completed_at = time.perf_counter()

    def summary(self) -> str:
        end = self.completed_at or time.perf_counter()
        total = end - self.started_at
        first = None if self.first_frame_at is None else self.first_frame_at - self.started_at
        receive = None if self.first_frame_at is None else end - self.first_frame_at
        return "\n".join(
            [
                "--- timing ---",
                f"captured frames: {self.captured_frames}",
                f"message id: {self.message_id or 'n/a'}",
                f"start to first QR recognition: {first:.3f} s" if first is not None else "start to first QR recognition: n/a",
                f"first QR recognition to full text output: {receive:.3f} s" if receive is not None else "first QR recognition to full text output: n/a",
                f"total client time: {total:.3f} s",
            ]
        )


def configure_acceleration(use_gpu: bool) -> str:
    global USE_GPU_ACCELERATION
    USE_GPU_ACCELERATION = use_gpu
    if not use_gpu:
        return "GPU acceleration: disabled"

    try:
        import cv2

        cuda_count = cv2.cuda.getCudaEnabledDeviceCount() if hasattr(cv2, "cuda") else 0
        opencl_available = cv2.ocl.haveOpenCL()
        cv2.ocl.setUseOpenCL(opencl_available)
        if cuda_count:
            return f"GPU acceleration: CUDA devices detected ({cuda_count}); QR fallback uses OpenCV acceleration where supported"
        if opencl_available:
            return "GPU acceleration: OpenCL enabled for OpenCV fallback preprocessing"
        return "GPU acceleration: no OpenCV CUDA/OpenCL device available; using optimized CPU path"
    except Exception as exc:
        return f"GPU acceleration: unavailable ({exc}); using optimized CPU path"


def read_text_file(path: Path) -> tuple[str, str]:
    data = path.read_bytes()
    for encoding in TEXT_ENCODINGS:
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace"), "utf-8-replace"


def decode_payloads_from_image(image: Image.Image) -> list[str]:
    payloads, _ = decode_payloads_and_bbox_from_image(image)
    return payloads


def _bbox_from_points(points: list[tuple[float, float]]) -> tuple[int, int, int, int] | None:
    if not points:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def _expand_screen_bbox(
    local_bbox: tuple[int, int, int, int],
    origin: tuple[int, int],
    screen_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    left, top, right, bottom = local_bbox
    width = max(1, right - left)
    height = max(1, bottom - top)
    padding = max(80, int(max(width, height) * 0.25))
    origin_x, origin_y = origin
    screen_width, screen_height = screen_size
    return (
        max(0, origin_x + left - padding),
        max(0, origin_y + top - padding),
        min(screen_width, origin_x + right + padding),
        min(screen_height, origin_y + bottom + padding),
    )


def decode_payloads_and_bbox_from_image(image: Image.Image) -> tuple[list[str], tuple[int, int, int, int] | None]:
    payloads: list[str] = []
    bbox: tuple[int, int, int, int] | None = None
    try:
        import zxingcpp

        for barcode in zxingcpp.read_barcodes(image.convert("RGB")):
            text = getattr(barcode, "text", "")
            if text:
                payloads.append(text)
                position = getattr(barcode, "position", None)
                if bbox is None and position is not None:
                    bbox = _bbox_from_points(
                        [
                            (position.top_left.x, position.top_left.y),
                            (position.top_right.x, position.top_right.y),
                            (position.bottom_right.x, position.bottom_right.y),
                            (position.bottom_left.x, position.bottom_left.y),
                        ]
                    )
    except Exception:
        pass
    if payloads:
        return list(dict.fromkeys(payloads)), bbox

    import cv2
    rgb = np.array(image.convert("RGB"))
    detector = cv2.QRCodeDetector()
    variants = [
        rgb,
        cv2.resize(rgb, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST),
    ]

    for variant in variants:
        if USE_GPU_ACCELERATION and cv2.ocl.useOpenCL():
            bgr = cv2.cvtColor(cv2.UMat(variant), cv2.COLOR_RGB2BGR).get()
        else:
            bgr = cv2.cvtColor(variant, cv2.COLOR_RGB2BGR)
        ok, decoded, _, _ = detector.detectAndDecodeMulti(bgr)
        if ok:
            payloads.extend(item for item in decoded if item)
            if bbox is None:
                detected, points = detector.detectMulti(bgr)
                if detected and points is not None and len(points):
                    bbox = _bbox_from_points([(float(x), float(y)) for x, y in points[0]])

        single, points, _ = detector.detectAndDecode(bgr)
        if single:
            payloads.append(single)
            if bbox is None and points is not None and len(points):
                bbox = _bbox_from_points([(float(x), float(y)) for x, y in points.reshape(-1, 2)])

    return list(dict.fromkeys(payloads)), bbox


def consume_payloads(
    payloads: list[str],
    frames: dict[tuple[str, int], Frame],
    stats: CaptureStats | None = None,
) -> str | None:
    changed = False
    completed_before = completed_message_ids(frames)
    for payload in payloads:
        try:
            frame = parse_payload(payload)
        except ProtocolError:
            continue

        key = (frame.message_id, frame.index)
        existing = frames.get(key)
        if existing is None:
            frames[key] = frame
            if stats is not None:
                stats.mark_frame(frame)
            changed = True
        elif existing.payload != frame.payload:
            frames[key] = frame
            changed = True
            print(
                f"replaced frame {frame.index + 1}/{frame.total} for {frame.message_id}; "
                "previous payload did not match latest decode",
                flush=True,
            )

    if changed:
        by_id: dict[str, set[int]] = {}
        totals: dict[str, int] = {}
        for frame in frames.values():
            by_id.setdefault(frame.message_id, set()).add(frame.index)
            totals[frame.message_id] = frame.total
        status = ", ".join(
            f"{msg_id}: {len(indexes)}/{totals[msg_id]}"
            for msg_id, indexes in sorted(by_id.items())
        )
        print(f"captured {status}", flush=True)

    text = assemble_frames(frames.values())
    if text is not None and stats is not None:
        if stats.completed_at is None:
            print("decode complete; CRC verified", flush=True)
        stats.mark_complete()
    elif changed:
        for message_id in sorted(completed_message_ids(frames) - completed_before):
            print(
                f"captured {message_id}: all frame indexes received, but CRC/decode failed; "
                "continuing to refresh frames from the next playback loop",
                flush=True,
            )
    return text


def capture_screen(
    interval: float,
) -> tuple[str, CaptureStats]:
    frames: dict[tuple[str, int], Frame] = {}
    stats = CaptureStats.start()
    bbox: tuple[int, int, int, int] | None = None
    screen_size: tuple[int, int] | None = None
    while True:
        image = ImageGrab.grab(bbox=bbox)
        if screen_size is None:
            screen_size = image.size

        if bbox is None:
            payloads, local_bbox = decode_payloads_and_bbox_from_image(image)
        else:
            payloads = decode_payloads_from_image(image)
            local_bbox = None

        if bbox is None and local_bbox is not None and screen_size is not None:
            origin = (bbox[0], bbox[1]) if bbox is not None else (0, 0)
            bbox = _expand_screen_bbox(local_bbox, origin, screen_size)
            print(f"auto screen crop: {bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}", flush=True)

        text = consume_payloads(payloads, frames, stats)
        if text is not None:
            return text, stats
        time.sleep(interval)


def capture_camera(camera_index: int, interval: float) -> tuple[str, CaptureStats]:
    import cv2
    frames: dict[tuple[str, int], Frame] = {}
    stats = CaptureStats.start()
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"could not open camera {camera_index}")

    try:
        while True:
            ok, image = capture.read()
            if not ok:
                time.sleep(interval)
                continue

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            payloads = decode_payloads_from_image(Image.fromarray(rgb))
            text = consume_payloads(payloads, frames, stats)
            if text is not None:
                return text, stats
            time.sleep(interval)
    finally:
        capture.release()


def decode_images(paths: list[Path]) -> tuple[str, CaptureStats]:
    frames: dict[tuple[str, int], Frame] = {}
    stats = CaptureStats.start()
    for path in paths:
        payloads = decode_payloads_from_image(Image.open(path))
        text = consume_payloads(payloads, frames, stats)
        if text is not None:
            return text, stats
    raise RuntimeError("not enough valid QR frames found in supplied images")


def decode_png_data_url(data_url: str) -> Image.Image:
    _, _, encoded = data_url.partition(",")
    if not encoded:
        encoded = data_url
    return Image.open(io.BytesIO(base64.b64decode(encoded)))


def simulate_from_server(url: str, text: str, chunk_size: int, playback_interval: float) -> tuple[str, CaptureStats]:
    stats = CaptureStats.start()
    body = json.dumps(
        {"text": text, "chunk_size": chunk_size, "include_png": True}
    ).encode("utf-8")
    req = urlrequest.Request(
        f"{url.rstrip('/')}/api/frames",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlrequest.urlopen(req) as response:
        payload = json.loads(response.read().decode("utf-8"))

    frames: dict[tuple[str, int], Frame] = {}
    for item in payload["frames"]:
        image = decode_png_data_url(item["png_data_url"])
        payloads = decode_payloads_from_image(image)
        text = consume_payloads(payloads, frames, stats)
        if text is not None:
            return text, stats
        time.sleep(playback_interval)
    raise RuntimeError("server returned incomplete frames")


def benchmark_local(char_count: int, chunk_size: int) -> tuple[str, CaptureStats, dict[str, float | int]]:
    text = ("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 " * ((char_count // 63) + 1))[:char_count]
    started_at = time.perf_counter()
    frames = split_text(text, chunk_size=chunk_size)
    pngs = [make_qr_png(frame.payload) for frame in frames]
    encoded_at = time.perf_counter()

    stats = CaptureStats.start()
    captured: dict[tuple[str, int], Frame] = {}
    result = None
    for png in pngs:
        payloads = decode_payloads_from_image(Image.open(__import__("io").BytesIO(png)))
        result = consume_payloads(payloads, captured, stats)
        if result is not None:
            break
    completed_at = time.perf_counter()
    metrics = {
        "chars": char_count,
        "frames": len(frames),
        "encode_ms": round((encoded_at - started_at) * 1000, 3),
        "decode_ms": round((completed_at - encoded_at) * 1000, 3),
        "end_to_end_ms": round((completed_at - started_at) * 1000, 3),
    }
    if result != text:
        raise RuntimeError("benchmark decode did not match source text")
    return result, stats, metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="QR text transfer client")
    parser.add_argument("--screen", action="store_true", help="capture QR frames from screen")
    parser.add_argument("--camera", type=int, help="capture QR frames from camera index")
    parser.add_argument("--image", nargs="*", type=Path, help="decode one or more saved QR PNGs")
    parser.add_argument("--simulate-url", help="server base URL for protocol simulation")
    parser.add_argument("--text-file", type=Path, help="text file used with --simulate-url")
    parser.add_argument("--benchmark-chars", type=int, help="run local QR generation/recognition benchmark")
    parser.add_argument("--use-gpu", action="store_true", help="enable OpenCV GPU/OpenCL acceleration when available")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--interval", type=float, default=DEFAULT_CAPTURE_INTERVAL)
    parser.add_argument("--playback-interval", type=float, default=DEFAULT_PLAYBACK_INTERVAL, help="simulated sender frame interval; default is 1/10 s")
    args = parser.parse_args()
    print(configure_acceleration(args.use_gpu), flush=True)

    try:
        if args.screen:
            text, stats = capture_screen(
                args.interval,
            )
        elif args.camera is not None:
            text, stats = capture_camera(args.camera, args.interval)
        elif args.image:
            text, stats = decode_images(args.image)
        elif args.simulate_url and args.text_file:
            source_text, encoding = read_text_file(args.text_file)
            print(f"text file encoding: {encoding}", flush=True)
            text, stats = simulate_from_server(
                args.simulate_url,
                source_text,
                args.chunk_size,
                args.playback_interval,
            )
        elif args.benchmark_chars:
            text, stats, metrics = benchmark_local(args.benchmark_chars, args.chunk_size)
            print("--- benchmark ---")
            for key, value in metrics.items():
                print(f"{key}: {value}")
        else:
            parser.error("choose --screen, --camera, --image, --benchmark-chars, or --simulate-url with --text-file")
            return 2
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print("\n--- reconstructed text ---")
    print(text)
    print()
    print(stats.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
