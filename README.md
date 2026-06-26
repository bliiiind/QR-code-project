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
The browser player defaults to 100 ms per frame, about 10 FPS. QR PNG generation
is published frame by frame, so display can start as soon as the first frame is
ready instead of waiting for the full sequence to finish encoding.

To use a text sample such as a `.txt` attachment, choose it with the text file
picker in the web UI and press Encode File. The server automatically tries
UTF-8, GB18030, GBK, and Big5 text decoding.

The web UI also supports browser speech-to-text input. Choose the speech
language, press Voice, speak into the browser microphone, then press Encode
after the recognized text appears in the message box. This uses the browser's
Web Speech API, which is available in Chrome/Edge on localhost.

## Run the client

Capture the QR player from the screen:

```powershell
.\.venv\Scripts\python client.py --screen
```

For faster screen simulation, the client starts with a full-screen capture.
After it recognizes a QR code for the first time, it automatically crops later
captures to the QR area and prints the selected crop coordinates.

Capture from a camera:

```powershell
.\.venv\Scripts\python client.py --camera 0
```

The client has no capture timeout. It keeps collecting until every QR frame is
received. It saves whichever frame is recognized first, then reconstructs the
message in frame-index order after all frames pass CRC validation. The default
capture interval is 1/60 s for screen/camera simulation.

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

The simulation decodes the server-generated PNG frames and uses a 1/10 s sender
interval by default, matching the browser playback rate.

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
  player defaults to about 10 FPS. Reduce chunk size if your camera has trouble
  focusing on dense QR images.
