from __future__ import annotations

import base64
import io
import math
import time
import zlib
from dataclasses import dataclass
from typing import Iterable


PROTOCOL = "QRTXT2"
LEGACY_PROTOCOL = "QRTXT1"
BASE45_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:"
DEFAULT_CHUNK_SIZE = 1600


@dataclass(frozen=True)
class Frame:
    message_id: str
    index: int
    total: int
    checksum: str
    codec: str
    chunk_code: str

    @property
    def payload(self) -> str:
        return ":".join(
            [
                PROTOCOL,
                self.message_id,
                str(self.index),
                str(self.total),
                self.checksum,
                self.codec,
                self.chunk_code,
            ]
        )


class ProtocolError(ValueError):
    pass


def checksum_for_text(text: str) -> str:
    return f"{zlib.crc32(text.encode('utf-8')) & 0xFFFFFFFF:08X}"


def make_message_id(text: str) -> str:
    crc = checksum_for_text(text)
    now = int(time.time() * 1000)
    return f"{crc[:8]}{now & 0xFFFF:04X}"


def base45_encode(data: bytes) -> str:
    result: list[str] = []
    for index in range(0, len(data), 2):
        if index + 1 < len(data):
            value = data[index] * 256 + data[index + 1]
            result.append(BASE45_ALPHABET[value % 45])
            result.append(BASE45_ALPHABET[(value // 45) % 45])
            result.append(BASE45_ALPHABET[value // 2025])
        else:
            value = data[index]
            result.append(BASE45_ALPHABET[value % 45])
            result.append(BASE45_ALPHABET[value // 45])
    return "".join(result)


def base45_decode(text: str) -> bytes:
    values = {char: index for index, char in enumerate(BASE45_ALPHABET)}
    result = bytearray()
    index = 0
    while index < len(text):
        if index + 2 < len(text):
            value = values[text[index]] + values[text[index + 1]] * 45 + values[text[index + 2]] * 2025
            result.append(value // 256)
            result.append(value % 256)
            index += 3
        elif index + 1 < len(text):
            value = values[text[index]] + values[text[index + 1]] * 45
            result.append(value)
            index += 2
        else:
            raise ProtocolError("invalid base45 length")
    return bytes(result)


def encode_message_bytes(data: bytes) -> tuple[str, str]:
    compressed = zlib.compress(data, level=1)
    if len(compressed) < len(data):
        return "Z", base45_encode(compressed)
    return "R", base45_encode(data)


def decode_message_bytes(codec: str, code: str) -> bytes:
    data = base45_decode(code)
    if codec == "Z":
        return zlib.decompress(data)
    if codec == "R":
        return data
    raise ProtocolError(f"unsupported codec {codec}")


def split_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> list[Frame]:
    if chunk_size < 64:
        raise ValueError("chunk_size must be at least 64 encoded characters")

    data = text.encode("utf-8")
    codec, encoded = encode_message_bytes(data)
    total = max(1, math.ceil(len(encoded) / chunk_size))
    message_id = make_message_id(text)
    checksum = checksum_for_text(text)

    frames: list[Frame] = []
    for index in range(total):
        start = index * chunk_size
        chunk_code = encoded[start : start + chunk_size]
        frames.append(Frame(message_id, index, total, checksum, codec, chunk_code))
    return frames


def parse_payload(payload: str) -> Frame:
    parts = payload.strip("\r\n").split(":", 6)
    if len(parts) == 7 and parts[0] == PROTOCOL:
        _, message_id, index, total, checksum, codec, chunk_code = parts
    elif len(parts) == 6 and parts[0] == LEGACY_PROTOCOL:
        _, message_id, index, total, checksum, chunk_code = parts
        codec = "B32"
    else:
        raise ProtocolError("not a qr text transfer frame")

    try:
        parsed_index = int(index)
        parsed_total = int(total)
    except ValueError as exc:
        raise ProtocolError("frame index/total is not numeric") from exc

    if parsed_total < 1 or not (0 <= parsed_index < parsed_total):
        raise ProtocolError("frame index is out of range")

    return Frame(message_id, parsed_index, parsed_total, checksum, codec, chunk_code)


def assemble_frames(frames: Iterable[Frame]) -> str | None:
    frames_by_id: dict[str, list[Frame]] = {}
    for frame in frames:
        frames_by_id.setdefault(frame.message_id, []).append(frame)

    for grouped in frames_by_id.values():
        total = grouped[0].total
        checksum = grouped[0].checksum
        codec = grouped[0].codec
        if len({frame.index for frame in grouped}) < total:
            continue
        if any(frame.total != total or frame.checksum != checksum or frame.codec != codec for frame in grouped):
            continue

        ordered = sorted(grouped, key=lambda frame: frame.index)[:total]
        try:
            code = "".join(frame.chunk_code for frame in ordered)
            if codec == "B32":
                padded = code + "=" * (-len(code) % 8)
                data = base64.b32decode(padded)
            else:
                data = decode_message_bytes(codec, code)
            text = data.decode("utf-8")
        except Exception:
            continue

        if checksum_for_text(text) == checksum:
            return text
    return None


def make_qr_png(payload: str, box_size: int = 8, border: int = 4) -> bytes:
    import qrcode
    from qrcode.constants import ERROR_CORRECT_L

    qr = qrcode.QRCode(
        version=None,
        error_correction=ERROR_CORRECT_L,
        box_size=box_size,
        border=border,
    )
    qr.add_data(payload, optimize=0)
    qr.make(fit=True)
    image = qr.make_image(fill_color="black", back_color="white").convert("RGB")

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def frame_to_dict(frame: Frame, include_png: bool = False) -> dict[str, str | int]:
    result: dict[str, str | int] = {
        "message_id": frame.message_id,
        "index": frame.index,
        "total": frame.total,
        "checksum": frame.checksum,
        "payload": frame.payload,
    }
    if include_png:
        png_b64 = base64.b64encode(make_qr_png(frame.payload)).decode("ascii")
        result["png_data_url"] = f"data:image/png;base64,{png_b64}"
    return result
