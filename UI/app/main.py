
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict
from io import BytesIO
from pydub import AudioSegment
import os
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import uuid, os
from .utils import save_upload_to_tempfile, run_detector, aggregate_results, confidence_from_prob

class AnalyzeResponse(BaseModel):
    job_id: Optional[str] = None
    score: float
    confidence: str 

# import your audio function
from UI.app.audio.app import analyze_audio_bytes

app = FastAPI(title="Audio AI Detector")

# Optional: allow your frontend to call backend if on different port
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Allowed audio extensions
ALLOWED_AUDIO_EXTS = ["mp3", "wav", "ogg", "flac", "m4a"]

def convert_to_wav(file_bytes: bytes, ext: str) -> bytes:
    """
    Convert uploaded audio to WAV if needed.
    Returns bytes of WAV file.
    """
    if ext.lower() == "wav":
        return file_bytes

    # Convert using pydub
    audio = AudioSegment.from_file(BytesIO(file_bytes), format=ext)
    out_buf = BytesIO()
    audio.export(out_buf, format="wav")
    return out_buf.getvalue()

async def analyze_audio(file: UploadFile = File(...)) -> Dict:
    # 1️⃣ Check extension
    ext = file.filename.split(".")[-1].lower()
    if ext not in ALLOWED_AUDIO_EXTS:
        raise HTTPException(status_code=400, detail="Unsupported audio format")

    # 2️⃣ Read bytes
    file_bytes = await file.read()

    # 3️⃣ Convert to WAV
    try:
        wav_bytes = convert_to_wav(file_bytes, ext)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Conversion to WAV failed: {e}")

    # 4️⃣ Call your audio analysis function
    try:
        # analyze_audio_bytes returns (probability, confidence)
        prob, conf,explanation = analyze_audio_bytes(wav_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio analysis failed: {e}")

    # 5️⃣ Return JSON
    return (prob,conf,explanation)
# app/main.py



@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...), detectors: Optional[str] = None):
    """
    Upload file (text, audio, or short video). Optional `detectors` csv param: detector1,detector2
    If omitted, runs all detectors found in app/detectors/.
    
    """

    ext = file.filename.split(".")[-1].lower()
    if ext in ALLOWED_AUDIO_EXTS:
        results = await analyze_audio(file)
    
    return AnalyzeResponse(score=results[0], confidence=results[1],explanation=results[2])