from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tempfile
import urllib.request

import numpy as np
import librosa
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import subprocess
import traceback

# Config
# Swap this later for a Dharug-fine-tuned checkpoint without changing code.
MODEL_NAME = os.environ.get(
    "MODEL_NAME",
    "jonatasgrosman/wav2vec2-large-xlsr-53-spanish"
)

# Maximum seconds to decode (prevents huge files from blocking the API)
MAX_AUDIO_SECONDS = int(os.environ.get("MAX_AUDIO_SECONDS", "30"))

# Device selection
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# App
app = Flask(__name__)
CORS(app)

processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()


# Helpers
def download_to_temp(url: str, suffix: str = ".wav") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    urllib.request.urlretrieve(url, path)
    return path


def load_audio_16k_mono(path: str) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(path, sr=16000, mono=True)
    if MAX_AUDIO_SECONDS and len(y) > MAX_AUDIO_SECONDS * 16000:
        y = y[: MAX_AUDIO_SECONDS * 16000]
    return y, 16000


def convert_audio(in_path: str, out_path: str):
    # Force decode any format (WebM/Opus, MP3, AAC, etc.) into 16kHz mono PCM WAV
    cmd = [
        "ffmpeg",
        "-y",               # overwrite
        "-i", in_path,      # input file
        "-ar", "16000",     # resample to 16kHz
        "-ac", "1",         # mono
        "-f", "wav",        # output format
        out_path
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

def get_stt_transcript(y: np.ndarray, sr: int) -> str:
    
    inputs = processor(y, sampling_rate=sr, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        logits = model(inputs.input_values.to(DEVICE)).logits
        
    pred_ids = torch.argmax(logits, dim=-1)
    text = processor.batch_decode(pred_ids)[0]
    
    return text.strip()

# Routes
@app.route("/transcribe", methods=["POST"])
def transcribe():
    try:
        payload = request.get_json(silent=True) or {}
        url = payload.get("url")
        if not url:
            return jsonify({"error": "Missing 'url'"}), 400

        # Download original file
        temp_in = download_to_temp(url)
        temp_out = temp_in + "_conv.wav"

        try:
            # Convert with ffmpeg 
            convert_audio(temp_in, temp_out)

            # Load with librosa
            y, sr = load_audio_16k_mono(temp_out)
            transcript = get_stt_transcript(y, sr)

        finally:
            for p in [temp_in, temp_out]:
                try: os.remove(p)
                except: pass

        return jsonify({
            "sttTranscript": transcript,
            "samplingRate": sr,
            "durationSec": round(len(y) / sr, 3),
            "model": MODEL_NAME
        }), 200

    except Exception as e:
        traceback.print_exc()
        # Ensure subprocess errors (from FFmpeg failure) are logged clearly
        if isinstance(e, subprocess.CalledProcessError):
            error_message = f"FFmpeg Error: {e.stderr.decode('utf-8')}"
            return jsonify({"error": error_message}), 500
        
        return jsonify({"error": str(e)}), 500

    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)