
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


from audio.app import analyze_audio_bytes

app = FastAPI(title="Audio AI Detector")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
def index():
    with open("./Frontend/index.html") as f:
        return f.read()

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

    ext = file.filename.split(".")[-1].lower()
    if ext not in ALLOWED_AUDIO_EXTS:
        raise HTTPException(status_code=400, detail="Unsupported audio format")

    
    file_bytes = await file.read()


    try:
        wav_bytes = convert_to_wav(file_bytes, ext)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Conversion to WAV failed: {e}")

    print("Converted to .wav")
    try:
        e =  analyze_audio_bytes(wav_bytes)
        print(e)
        prob, conf,explanation = e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio analysis failed: {e}")

    
    return (prob,conf,explanation)

async def text


@app.post("/analyze")
async def analyze(file: UploadFile = File(...), detectors: Optional[str] = None):
    """
    Upload file (text, audio, or short video). Optional `detectors` csv param: detector1,detector2
    If omitted, runs all detectors found in app/detectors/.
    
    """
    try:
        ext = file.filename.split(".")[-1].lower()
        if ext in ALLOWED_AUDIO_EXTS:
            results = await analyze_audio(file)
        if ext in ALLOWED_VIDEO_EXTS:
            results = await analyze_video(file)
        if ext in ALLOWED_TEXT_EXTS:
            results = await analyze_text(file)
        return AnalyzeResponse(score=results[0], confidence=results[1],explanation=results[2])
    except Exception as e: 
        print(f"Exception :{e}")

