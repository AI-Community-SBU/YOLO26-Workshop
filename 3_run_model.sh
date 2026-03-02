#!/usr/bin/env bash
# Run the webcam inference app
# In Codespaces: the port will auto-forward — just click the link!
set -e
cd "$(dirname "$0")"
pip install -q --no-cache-dir -r requirements.txt -c constraints.txt 2>/dev/null
echo ""
echo "🚀 Starting RPS Detector (HTTPS)..."
echo "   Open https://localhost:8000 in your browser."
echo "   ⚠️  Click 'Advanced' → 'Proceed' to accept the self-signed cert."
echo ""
python3 run_webcam.py
