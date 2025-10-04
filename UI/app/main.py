# app/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import uuid, os
from .utils import save_upload_to_tempfile, run_detector, aggregate_results, confidence_from_prob

app = FastAPI(title="AI-Generation Detector Aggregator")

class DetectorResult(BaseModel):
    detector: str
    probability: float
    confidence: str
    explanation: str
    raw: dict

class AnalyzeResponse(BaseModel):
    job_id: Optional[str] = None
    results: List[DetectorResult]
    aggregated_score: float
    aggregated_confidence: str

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...), detectors: Optional[str] = None):
    """
    Upload file (text, audio, or short video). Optional `detectors` csv param: detector1,detector2
    If omitted, runs all detectors found in app/detectors/.
    """
    # Save upload to temp file
    try:
        input_path = await save_upload_to_tempfile(file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to save upload: {e}")

    try:
        results_raw = run_detector(input_path, detectors_csv=detectors)
    finally:
        # cleanup local file
        try:
            os.remove(input_path)
        except Exception:
            pass

    results = []
    for r in results_raw:
        prob = float(r.get("probability", 0.0))
        conf = confidence_from_prob(prob)
        res = DetectorResult(
            detector=r.get("detector", "unknown"),
            probability=prob,
            confidence=conf,
            explanation=r.get("explanation", ""),
            raw=r.get("raw", {})
        )
        results.append(res)

    aggregated_score = aggregate_results([r.probability for r in results])
    aggregated_confidence = confidence_from_prob(aggregated_score)

    return AnalyzeResponse(results=results, aggregated_score=aggregated_score, aggregated_confidence=aggregated_confidence)

