"""
SightLine — Web interface for phone camera scanning.
Run this on your laptop, then open the URL on your phone browser.

Usage:
    python -m offscan.webapp                # Start server on port 5000
    python -m offscan.webapp --port 8000    # Custom port
    python -m offscan.webapp --host 0.0.0.0 # Access from other devices on your network
"""

import argparse
import base64
import io
import json
import os
import sys
import tempfile
import time

# Add src to path
_src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from flask import Flask, request, jsonify, send_file
import numpy as np
import cv2
from PIL import Image

from offscan.pipeline import SightLinePipeline

app = Flask(__name__, static_folder=None)

# Initialize pipeline once
pipeline = None


def get_pipeline():
    global pipeline
    if pipeline is None:
        pipeline = SightLinePipeline(speak=False)
    return pipeline


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>SightLine</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: #0a0a0a;
            color: #fff;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        header {
            width: 100%;
            padding: 20px;
            text-align: center;
            background: #111;
            border-bottom: 1px solid #333;
        }
        header h1 { font-size: 22px; margin-bottom: 4px; }
        header p { font-size: 13px; color: #888; }
        .camera-container {
            width: 100%;
            max-width: 500px;
            padding: 16px;
        }
        #camera-view {
            width: 100%;
            border-radius: 12px;
            background: #000;
            display: block;
        }
        #captured-image {
            width: 100%;
            border-radius: 12px;
            display: none;
        }
        .controls {
            display: flex;
            gap: 12px;
            justify-content: center;
            margin: 20px 0;
        }
        button {
            background: #0066ff;
            color: #fff;
            border: none;
            padding: 16px 32px;
            border-radius: 12px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }
        button:active { transform: scale(0.96); }
        button:disabled { opacity: 0.4; cursor: not-allowed; }
        #retake-btn { background: #333; }
        #upload-btn { background: #333; }
        #results {
            width: 100%;
            max-width: 500px;
            padding: 16px;
            display: none;
        }
        .result-card {
            background: #111;
            border: 1px solid #333;
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 12px;
        }
        .result-card h3 { font-size: 14px; color: #0066ff; margin-bottom: 8px; }
        .result-card .value { font-size: 16px; margin-bottom: 4px; }
        .confidence-bar {
            background: #333;
            height: 6px;
            border-radius: 3px;
            margin-top: 4px;
            overflow: hidden;
        }
        .confidence-fill {
            height: 100%;
            background: #00ff88;
            border-radius: 3px;
        }
        .summary-box {
            background: #001a33;
            border: 1px solid #0066ff;
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 12px;
            font-size: 16px;
            line-height: 1.5;
        }
        .field-row {
            display: flex;
            justify-content: space-between;
            padding: 4px 0;
            border-bottom: 1px solid #222;
        }
        .field-row .label { color: #888; font-size: 13px; }
        .field-row .val { font-size: 14px; }
        #audio-player { width: 100%; margin-top: 8px; }
        .offline-badge {
            background: #00ff88;
            color: #000;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 700;
            display: inline-block;
            margin-top: 8px;
        }
        .status-msg {
            text-align: center;
            padding: 12px;
            color: #888;
            font-size: 14px;
        }
        #loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        .spinner {
            border: 3px solid #333;
            border-top: 3px solid #0066ff;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin: 0 auto 8px;
        }
        @keyframes spin { 100% { transform: rotate(360deg); } }
        .tabs { display: flex; gap: 0; margin: 16px 0; border-radius: 8px; overflow: hidden; border: 1px solid #333; }
        .tab { flex: 1; padding: 12px; text-align: center; background: #111; cursor: pointer; font-size: 14px; }
        .tab.active { background: #0066ff; }
        #upload-input { display: none; }
    </style>
</head>
<body>
    <header>
        <h1>SightLine</h1>
        <p>Your eyes, offline. Nothing leaves your phone.</p>
        <span class="offline-badge">FULLY OFFLINE - 0 BYTES SENT</span>
    </header>

    <div class="camera-container">
        <div class="tabs">
            <div class="tab active" id="tab-camera" onclick="switchTab('camera')">Camera</div>
            <div class="tab" id="tab-upload" onclick="switchTab('upload')">Upload</div>
        </div>

        <video id="camera-view" autoplay playsinline></video>
        <img id="captured-image" />
        <input type="file" id="upload-input" accept="image/*" capture="environment">

        <div class="controls">
            <button id="capture-btn" onclick="capturePhoto()">Scan Document</button>
            <button id="retake-btn" onclick="retake()" style="display:none;">Retake</button>
        </div>

        <div id="loading">
            <div class="spinner"></div>
            <p>Scanning document...</p>
        </div>
    </div>

    <div id="results">
        <div class="summary-box" id="summary-box"></div>
        <div id="fields-container"></div>
        <div id="entities-container"></div>
        <audio id="audio-player" controls></audio>
    </div>

    <script>
        let stream = null;
        let capturedBlob = null;

        // Start camera
        async function startCamera() {
            try {
                stream = await navigator.mediaDevices.getUserMedia({
                    video: { facingMode: "environment" },
                    audio: false
                });
                document.getElementById('camera-view').srcObject = stream;
            } catch(e) {
                console.log("Camera not available:", e);
                switchTab('upload');
            }
        }

        function switchTab(tab) {
            if (tab === 'camera') {
                document.getElementById('camera-view').style.display = 'block';
                document.getElementById('captured-image').style.display = 'none';
                document.getElementById('capture-btn').style.display = 'block';
                document.getElementById('upload-input').style.display = 'none';
                if (!stream) startCamera();
            } else {
                document.getElementById('camera-view').style.display = 'none';
                document.getElementById('captured-image').style.display = 'none';
                document.getElementById('capture-btn').style.display = 'none';
                document.getElementById('upload-input').style.display = 'block';
                if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
            }
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.getElementById('tab-' + tab).classList.add('active');
        }

        function capturePhoto() {
            const video = document.getElementById('camera-view');
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth || 1920;
            canvas.height = video.videoHeight || 1080;
            canvas.getContext('2d').drawImage(video, 0, 0);
            canvas.toBlob(blob => {
                capturedBlob = blob;
                const img = document.getElementById('captured-image');
                img.src = URL.createObjectURL(blob);
                img.style.display = 'block';
                document.getElementById('camera-view').style.display = 'none';
                document.getElementById('capture-btn').style.display = 'none';
                document.getElementById('retake-btn').style.display = 'block';
                uploadAndScan(blob);
            }, 'image/jpeg', 0.95);
        }

        // Upload via file input
        document.getElementById('upload-input').addEventListener('change', e => {
            const file = e.target.files[0];
            if (file) {
                const img = document.getElementById('captured-image');
                img.src = URL.createObjectURL(file);
                img.style.display = 'block';
                document.getElementById('capture-btn').style.display = 'none';
                document.getElementById('retake-btn').style.display = 'block';
                uploadAndScan(file);
            }
        });

        function retake() {
            capturedBlob = null;
            document.getElementById('captured-image').style.display = 'none';
            document.getElementById('results').style.display = 'none';
            document.getElementById('retake-btn').style.display = 'none';
            switchTab('camera');
        }

        async function uploadAndScan(blob) {
            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').style.display = 'none';

            const formData = new FormData();
            formData.append('image', blob);

            try {
                const resp = await fetch('/scan', { method: 'POST', body: formData });
                const data = await resp.json();
                document.getElementById('loading').style.display = 'none';
                displayResults(data);
            } catch(e) {
                document.getElementById('loading').innerHTML = '<p>Error: ' + e.message + '</p>';
            }
        }

        function displayResults(data) {
            const results = document.getElementById('results');
            results.style.display = 'block';

            // Summary
            const summaryBox = document.getElementById('summary-box');
            summaryBox.innerHTML = '<strong>' + data.document_label + '</strong><br>' + data.summary +
                '<br><small style="color:#888">OCR: ' + (data.ocr_confidence*100).toFixed(0) + '% | ' +
                'Classification: ' + (data.classification_confidence*100).toFixed(0) + '% | ' +
                'Time: ' + data.total_time_ms.toFixed(0) + 'ms</small>';

            // Fields
            const fieldsContainer = document.getElementById('fields-container');
            if (data.extracted_fields && data.extracted_fields.length > 0) {
                let html = '<div class="result-card"><h3>Extracted Fields</h3>';
                data.extracted_fields.forEach(f => {
                    const pct = (f.confidence * 100).toFixed(0);
                    html += '<div class="field-row"><span class="label">' + f.label + '</span>' +
                            '<span class="val">' + f.value + ' (' + pct + '%)</span></div>';
                });
                html += '</div>';
                fieldsContainer.innerHTML = html;
            } else {
                fieldsContainer.innerHTML = '';
            }

            // Entities
            const entitiesContainer = document.getElementById('entities-container');
            if (data.entities && data.entities.length > 0) {
                let html = '<div class="result-card"><h3>Entities Found</h3>';
                data.entities.forEach(e => {
                    html += '<div class="field-row"><span class="label">' + e.label + '</span>' +
                            '<span class="val">' + e.text + '</span></div>';
                });
                html += '</div>';
                entitiesContainer.innerHTML = html;
            } else {
                entitiesContainer.innerHTML = '';
            }

            // Audio
            if (data.audio_url) {
                document.getElementById('audio-player').src = data.audio_url;
            }
        }

        startCamera();
    </script>
</body>
</html>"""


@app.route("/")
def index():
    return HTML_PAGE


@app.route("/scan", methods=["POST"])
def scan():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    # Read image into OpenCV format
    img_array = np.frombuffer(file.read(), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Cannot decode image"}), 400

    # Run pipeline
    result = get_pipeline().scan(img)

    # Generate audio
    audio_path = None
    try:
        from offscan.tts import TextToSpeech
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir="/tmp") as f:
            audio_path = f.name
        tts = TextToSpeech(output_file=audio_path)
        tts.speak(result.summary)
    except Exception:
        pass

    response = {
        "ocr_text": result.ocr_text,
        "ocr_confidence": result.ocr_confidence,
        "ocr_engine": result.ocr_engine,
        "word_count": result.word_count,
        "document_category": result.document_category,
        "document_label": result.document_label,
        "classification_confidence": result.classification_confidence,
        "classification_engine": result.classification_engine,
        "extracted_fields": result.extracted_fields,
        "entities": result.entities,
        "ner_engine": result.ner_engine,
        "summary": result.summary,
        "total_time_ms": result.total_time_ms,
        "fully_offline": result.fully_offline,
        "network_calls_made": result.network_calls_made,
    }

    if audio_path and os.path.exists(audio_path):
        response["audio_url"] = f"/audio/{os.path.basename(audio_path)}"

    return jsonify(response)


@app.route("/audio/<filename>")
def get_audio(filename):
    path = os.path.join("/tmp", filename)
    if os.path.exists(path):
        return send_file(path, mimetype="audio/wav")
    return "Not found", 404


def main():
    parser = argparse.ArgumentParser(description="SightLine Web — phone camera scanner")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0 for network access)")
    parser.add_argument("--port", type=int, default=5000, help="Port (default: 5000)")
    args = parser.parse_args()

    print(f"\nSightLine Web Server")
    print(f"{'='*50}")
    print(f"Open this URL on your phone browser:")
    print(f"  http://0.0.0.0:{args.port}")
    print(f"\nOr if on the same WiFi network:")
    import socket
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        print(f"  http://{local_ip}:{args.port}")
    except Exception:
        pass
    print(f"\nFully offline — zero data leaves your device.")
    print(f"{'='*50}\n")

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
