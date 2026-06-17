# QR Text Transfer

This project simulates one-way text transfer with QR images.

The server splits long text into ordered QR frames and plays them in a loop.
The client captures frames from a camera, the screen, or saved images, then
reassembles the original text after all frames pass the CRC32 check.

## Setup

Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

Linux server:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip v4l-utils

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

## Run the server

Windows:

```powershell
.\.venv\Scripts\python app.py
```

Linux/headless server:

```bash
source .venv/bin/activate
python app.py --host 0.0.0.0 --port 5000
```

For systemd or non-interactive services, disable the terminal input loop:

```bash
python app.py --host 0.0.0.0 --port 5000 --no-terminal-input
```

Open http://SERVER_IP:5000 from another machine to enter text in the web UI.

Type or paste text in the web UI and press Encode. You can also type text in
the server terminal and press Enter. Both paths convert the text into QR frames
immediately; the browser page automatically plays the latest QR sequence.

On a fully headless server, submit text through the API instead of a browser:

```bash
curl -X POST http://127.0.0.1:5000/api/message \
  -H 'Content-Type: application/json' \
  -d '{"text":"hello from linux","chunk_size":1600}'
```

## Drive a MicroLED display on Linux

The server exposes QR frames without requiring a desktop browser:

- `GET /api/display/next.png` returns the next QR frame as PNG and advances the playback cursor.
- `GET /api/frame/<index>.png` returns a specific frame from the current message.
- Add `size=512`, `box_size=8`, or `border=4` query parameters to fit your panel.

If your MicroLED vendor provides a command that can draw a PNG, run the generic
puller and pass that command with `{image}` as the current frame path:

```bash
source .venv/bin/activate
python microled_player.py \
  --server http://127.0.0.1:5000 \
  --size 512 \
  --interval-ms 80 \
  --command 'microled-show --image {image}'
```

If your display program watches a file instead, omit `--command`; the latest QR
frame is continuously written to `/tmp/qr-microled-frame.png` by default:

```bash
python microled_player.py --server http://127.0.0.1:5000 --size 512
```

## Run the client

Capture the QR player from the screen:

```powershell
.\.venv\Scripts\python client.py --screen
```

Capture from a camera:

```powershell
.\.venv\Scripts\python client.py --camera 0
```

Linux camera capture:

```bash
v4l2-ctl --list-devices
source .venv/bin/activate
python client.py --camera /dev/video0 --camera-width 1280 --camera-height 720 --camera-fps 30
```

Enable GPU/OpenCL fallback acceleration when the local OpenCV build and device support it:

```powershell
.\.venv\Scripts\python client.py --screen --use-gpu
```

The client prints the reconstructed text and timing information:

- `start to first QR recognition`: time from client start to the first valid QR frame.
- `first QR recognition to full text output`: time from the first valid QR frame to complete text reconstruction.
- `total client time`: full elapsed time on the client side.

Protocol-only simulation through the server API:

```powershell
.\.venv\Scripts\python client.py --simulate-url http://127.0.0.1:5000 --text-file sample.txt
```

Local 1000-character benchmark:

```powershell
.\.venv\Scripts\python client.py --benchmark-chars 1000 --use-gpu
```

## Notes

- Each QR payload uses `QRTXT2:message_id:index:total:crc32:codec:base45_chunk`.
  The server compresses the full message with fast zlib when it helps, then
  Base45-encodes it into QR alphanumeric-mode characters for smaller, faster
  QR frames.
- If a frame is missed, the client keeps listening until it appears in the next playback loop.
- The default chunk size is 1600 encoded characters per QR image and the web
  player defaults to 80 ms per frame. This is tuned for the 1000-character
  under-1-second target; reduce chunk size if your camera has trouble focusing
  on dense QR images.
