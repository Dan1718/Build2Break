# detectors.py
import numpy as np
from models.frame_detector import FrameDetector
from models.temporal_detector import TemporalDetector
from utils import calculate_confidence

class VideoDetector:
    """
    High-level orchestrator that runs both spatial (frame-based) and
    temporal (motion-based) detection to compute ai_probability, confidence,
    and a short human-readable explanation.
    """
    def __init__(self, frame_model_path="models/xception_weights.pth",
                       raft_weights_path="models/raft-sintel.pth",
                       device=None):
        self.device = device
        # Initialize detectors
        self.frame_detector = FrameDetector(weights_path=frame_model_path, device=device)
        self.temporal_detector = TemporalDetector(weights_path=raft_weights_path, device=device)

    def run_detection(self, frames):
        """
        Runs both frame and temporal detectors and combines results.
        Returns:
            dict with keys:
            - ai_probability: float [0,1]
            - confidence: float [0,1]
            - explanation: str
            - frame_score: raw frame detector probability
            - temporal_score: raw temporal anomaly score
        """
        # Run frame detector
        try:
            frame_score = self.frame_detector.detect(frames)
        except Exception as e:
            frame_score = 0.0
            print(f"[VideoDetector] Frame detector failed: {e}")

        # Run temporal detector
        try:
            temporal_score, magnitudes = self.temporal_detector.detect(frames)
        except Exception as e:
            temporal_score = 0.0
            magnitudes = []
            print(f"[VideoDetector] Temporal detector failed: {e}")

        # Combine scores (weighted average)
        # Here you can adjust weights depending on your preference
        ai_probability = float(np.clip(0.6 * frame_score + 0.4 * temporal_score, 0.0, 1.0))

        # Confidence based on difference and temporal consistency
        confidence = calculate_confidence(frame_score, temporal_score, magnitudes)

        # Human-readable explanation
        explanation_parts = []
        explanation_parts.append(f"The frame-based detector estimated {frame_score:.2f} probability of AI-generated content.")
        explanation_parts.append(f"The temporal consistency detector estimated {temporal_score:.2f} probability based on motion anomalies.")
        explanation_parts.append(f"The combined AI probability is {ai_probability:.2f}, with confidence {confidence:.2f}.")
        if magnitudes:
            explanation_parts.append(f"Average optical flow magnitude across frames: {np.mean(magnitudes):.3f}")

        explanation = " ".join(explanation_parts)

        return {
            "ai_probability": ai_probability,
            "confidence": confidence,
            "frame_score": frame_score,
            "temporal_score": temporal_score,
            "explanation": explanation
        }
