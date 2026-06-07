# QR Text Transfer

This project simulates one-way text transfer with QR images.

The server splits long text into ordered QR frames and plays them in a loop.
The client captures frames from a camera, the screen, or saved images, then
reassembles the original text after all frames pass the CRC32 check.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

## Run the server

```powershell
.\.venv\Scripts\python app.py
```

Open http://127.0.0.1:5000 to display the QR player and enter text in the web UI.

Type or paste text in the web UI and press Encode. You can also type text in
the server terminal and press Enter. Both paths convert the text into QR frames
immediately; the browser page automatically plays the latest QR sequence.

## Run the client

Capture the QR player from the screen:

```powershell
.\.venv\Scripts\python client.py --screen
```

Capture from a camera:

```powershell
.\.venv\Scripts\python client.py --camera 0
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
