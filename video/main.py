# app.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import time
import tempfile
import shutil
import os

from video.preprocessor import VideoPreprocessor
from video.detectors import VideoDetector

app = FastAPI(title="AI Video Detector")

@app.post("/analyze_video")
async def detect_video(file: UploadFile = File(...)):
    """
    Upload a single video file and get AI detection results.
    """
    # Validate file type
    if not file.filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
        return JSONResponse(
            status_code=400,
            content={"status": "error", "details": "Unsupported file type"}
        )
    
    start_time = time.time()
    output = {"video_file": file.filename, "status": "error", "details": "", "processing_time": None}

    try:
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name

        # Preprocess video
        pre = VideoPreprocessor()
        res, msg = pre.process(tmp_path)
        if res is None:
            output.update({
                "status": "error",
                "ai_probability":1,
                "confidence":1,
                "explanation":"1",
                "details": msg,
                "processing_time": round(time.time() - start_time, 3)
            })
            return output

        frames, _ = res

        # Run detector
        detector = VideoDetector()
        result = detector.run_detection(frames)
        print(output)
        output.update({
            "status": "success",
            "ai_probability": result["ai_probability"],
            "confidence": result["confidence"],
            "explanation": result.get("explanation", ""),
            "processing_time": round(time.time() - start_time, 3)
        })

    except Exception as e:
        output.update({
            "status": "error",
            "details": str(e),
            "processing_time": round(time.time() - start_time, 3)
        })

    finally:
        # Clean up temp file
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)

    return output
