from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from urllib import error as urlerror
from urllib import request as urlrequest


def fetch_frame(url: str, output: Path) -> dict[str, str]:
    with urlrequest.urlopen(url, timeout=10) as response:
        data = response.read()
        headers = {key: value for key, value in response.headers.items()}

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(data)
    return headers


def run_display_command(command: str, image: Path, headers: dict[str, str]) -> None:
    substitutions = {
        "image": str(image),
        "version": headers.get("X-QR-Version", ""),
        "index": headers.get("X-QR-Frame-Index", ""),
        "total": headers.get("X-QR-Frame-Total", ""),
        "checksum": headers.get("X-QR-Checksum", ""),
    }
    subprocess.run(command.format(**substitutions), shell=True, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Pull QR frames and hand them to a MicroLED display command")
    parser.add_argument("--server", default="http://127.0.0.1:5000", help="QR server base URL")
    parser.add_argument("--output", type=Path, default=Path("/tmp/qr-microled-frame.png"), help="where to save the current frame PNG")
    parser.add_argument("--interval-ms", type=int, default=80, help="delay between frame pulls")
    parser.add_argument("--size", type=int, help="resize QR PNG to a square pixel size before display")
    parser.add_argument("--box-size", type=int, default=8, help="QR module pixel size when not using --size")
    parser.add_argument("--border", type=int, default=4, help="QR quiet-zone modules")
    parser.add_argument(
        "--command",
        help=(
            "MicroLED display command. Placeholders: {image}, {version}, {index}, {total}, {checksum}. "
            "Example: 'microled-show --image {image}'"
        ),
    )
    parser.add_argument("--once", action="store_true", help="fetch and display only one frame")
    args = parser.parse_args()

    query = f"interval_ms={args.interval_ms}&box_size={args.box_size}&border={args.border}"
    if args.size:
        query += f"&size={args.size}"
    url = f"{args.server.rstrip('/')}/api/display/next.png?{query}"

    while True:
        try:
            headers = fetch_frame(url, args.output)
            if args.command:
                run_display_command(args.command, args.output, headers)
            print(
                "displayed frame "
                f"{int(headers.get('X-QR-Frame-Index', '0')) + 1}/"
                f"{headers.get('X-QR-Frame-Total', '?')} "
                f"version={headers.get('X-QR-Version', '?')}",
                flush=True,
            )
        except urlerror.HTTPError as exc:
            if exc.code == 404:
                print("waiting for QR message...", flush=True)
            else:
                print(f"fetch error: HTTP {exc.code}", file=sys.stderr, flush=True)
        except Exception as exc:
            print(f"display error: {exc}", file=sys.stderr, flush=True)

        if args.once:
            return 0
        time.sleep(max(0.02, args.interval_ms / 1000))


if __name__ == "__main__":
    raise SystemExit(main())
