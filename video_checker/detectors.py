# detectors.py
import time
from models.frame_detector import FrameDetector
from models.temporal_detector import TemporalDetector
from utils import compute_confidence_from_scores, generate_explanation

class VideoDetector:
    """
    High-level orchestrator:
      - runs frame detector (only used as additional signal when faces present)
      - runs temporal detector on all videos
      - combines scores conservatively
    """

    def __init__(self, face_sensitive=True):
        self.frame_detector = FrameDetector()
        self.temporal_detector = TemporalDetector()
        self.face_sensitive = bool(face_sensitive)

    def run_detection(self, frames):
        t0 = time.time()
        # Frame detector (may be skipped if no faces)
        try:
            frame_score, face_present = self.frame_detector.detect(frames, face_sensitive=self.face_sensitive)
        except Exception:
            frame_score, face_present = 0.0, False

        # Temporal detector (always run)
        try:
            temporal_score, magnitudes = self.temporal_detector.detect(frames)
        except Exception:
            temporal_score, magnitudes = 0.0, []

        # Combine:
        if face_present:
            ai_probability = float(((frame_score or 0.0) + (temporal_score or 0.0)) / 2.0)
        else:
            # When no faces, frame detector is not meaningful. Use temporal only.
            ai_probability = float(temporal_score or 0.0)
            # Set frame_score to None to indicate unused (so confidence becomes 1.0 in compute_confidence)
            frame_score = None

        confidence = compute_confidence_from_scores(frame_score, temporal_score)
        explanation = generate_explanation(frame_score if frame_score is not None else 0.0,
                                           temporal_score, confidence, face_present)

        return {
            "ai_probability": float(ai_probability),
            "confidence": float(confidence),
            "frame_score": (float(frame_score) if frame_score is not None else None),
            "temporal_score": float(temporal_score),
            "face_present": bool(face_present),
            "magnitudes": magnitudes,
            "explanation": explanation,
            "detection_time_sec": round(time.time() - t0, 3)
        }
