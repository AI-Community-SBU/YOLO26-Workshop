#!/usr/bin/env python3
"""
Rock-Paper-Scissors YOLO Webcam App
Run in Codespaces → opens in browser → uses your webcam → runs inference on server.
"""

import asyncio
import base64
import json
import os
import ssl
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from ultralytics import YOLO

# ── Config ──────────────────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", str(Path(__file__).parent / "pretrained_weights" / "best.pt"))
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.5"))
PORT = int(os.getenv("PORT", "8000"))

# ── Load model once at startup ──────────────────────────────────────────
print(f"⏳ Loading model from {MODEL_PATH} ...")
model = YOLO(MODEL_PATH)
CLASS_NAMES = model.names  # {0: 'Paper', 1: 'Rock', 2: 'Scissors'}
print(f"✅ Model loaded! Classes: {CLASS_NAMES}")

app = FastAPI()

# ── Serve the HTML page ─────────────────────────────────────────────────
@app.get("/")
async def root():
    html_path = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(html_path.read_text())


# ── WebSocket: receive frames, return detections ────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("🔌 Client connected")
    try:
        while True:
            # Receive base64 JPEG from browser
            data = await ws.receive_text()
            img_bytes = base64.b64decode(data)
            img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

            if frame is None:
                await ws.send_text(json.dumps({"detections": []}))
                continue

            # Run YOLO
            results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)
            detections = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append({
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "class": CLASS_NAMES[int(box.cls[0])],
                        "conf": round(float(box.conf[0]), 2),
                    })

            await ws.send_text(json.dumps({"detections": detections}))

    except WebSocketDisconnect:
        print("🔌 Client disconnected")
    except Exception as e:
        print(f"❌ Error: {e}")


# ── Run ─────────────────────────────────────────────────────────────────
def _ensure_ssl_cert():
    """Auto-generate a self-signed cert so the browser allows camera access."""
    cert_dir = Path(__file__).parent / ".certs"
    cert_file = cert_dir / "cert.pem"
    key_file = cert_dir / "key.pem"
    if cert_file.exists() and key_file.exists():
        return str(cert_file), str(key_file)
    cert_dir.mkdir(exist_ok=True)
    print("🔐 Generating self-signed SSL certificate (one-time)...")
    subprocess.run([
        "openssl", "req", "-x509", "-newkey", "rsa:2048",
        "-keyout", str(key_file), "-out", str(cert_file),
        "-days", "365", "-nodes",
        "-subj", "/CN=localhost",
    ], check=True, capture_output=True)
    print("✅ SSL cert ready")
    return str(cert_file), str(key_file)


if __name__ == "__main__":
    import uvicorn
    cert, key = _ensure_ssl_cert()
    print(f"\n🚀 Starting HTTPS server on port {PORT}")
    print(f"   Open:  https://localhost:{PORT}")
    print(f"   ⚠️  Your browser will warn about the self-signed cert — click 'Advanced' → 'Proceed'.\n")
    uvicorn.run(app, host="0.0.0.0", port=PORT, ssl_certfile=cert, ssl_keyfile=key)
