# detectors.py
import time
from video.models.frame_detector import FrameDetector
from video.models.temporal_detector import TemporalDetector
from video.utils import compute_confidence_from_scores, generate_explanation

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

        # --- Weighted combination ---
        if face_present:
            # Give slightly higher weight to frame_score
            combined_score = 0.65 * frame_score + 0.35 * temporal_score
        else:
            combined_score = temporal_score  # only temporal

    # --- Non-linear scaling for decisiveness ---
    # Push values closer to 0 or 1
    def sigmoid(x, steep=10):
        import math
        return 1 / (1 + math.exp(-steep * (x - 0.5)))

        ai_probability = sigmoid(combined_score, steep=12)  # sharper

        # --- Magnitude tweak ---
        if magnitudes:
            mag_factor = min(sum(magnitudes)/len(magnitudes), 1.0)  # avg magnitude capped at 1
            ai_probability = ai_probability * 0.7 + mag_factor * 0.3  # blend in magnitude info

        # Frame_score unused when no faces
        frame_score_for_conf = frame_score if frame_score is not None else 0.0

        # --- Confidence adjustment ---
        # Higher if frame & temporal agree AND probability far from 0.5
        agreement = 1 - abs((frame_score_for_conf or 0.0) - temporal_score)
        magnitude_strength = max(frame_score_for_conf or 0.0, temporal_score)
        confidence = 0.5 * agreement + 0.5 * magnitude_strength
        # Boost slightly if probability is near edges (decisive)
        confidence += 0.1 * (abs(ai_probability - 0.5) * 2)
        confidence = min(max(confidence, 0.0), 1.0)  # clamp 0-1

        explanation = generate_explanation(frame_score_for_conf,
                                        temporal_score, confidence, face_present)

        return {
            "ai_probability": float(round(ai_probability, 3)),
            "confidence": float(round(confidence, 3)),
            "frame_score": (float(frame_score) if frame_score is not None else None),
            "temporal_score": float(temporal_score),
            "face_present": bool(face_present),
            "magnitudes": magnitudes,
            "explanation": explanation,
            "detection_time_sec": round(time.time() - t0, 3)
        }

