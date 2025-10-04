# utils.py
import math

def compute_confidence_from_scores(frame_score, temporal_score):
    """
    Per requirement: confidence = 1.0 - abs(frame_score - temporal_score)
    Both inputs expected in [0,1]. If frame_score is None (unused), we'll set it equal
    to temporal_score so confidence becomes 1.0 (max confidence when only temporal is used).
    """
    try:
        if frame_score is None:
            frame_score = temporal_score
        frame_score = float(frame_score or 0.0)
        temporal_score = float(temporal_score or 0.0)
        diff = abs(frame_score - temporal_score)
        conf = 1.0 - diff
        # Clip to [0,1]
        if conf < 0.0:
            conf = 0.0
        if conf > 1.0:
            conf = 1.0
        return conf
    except Exception:
        return 0.0

def generate_explanation(frame_score, temporal_score, confidence, face_present):
    """
    Short human-readable explanation of result.
    """
    try:
        if face_present:
            ai_prob = (frame_score + temporal_score) / 2.0
            source = "combined frame+temporal"
        else:
            ai_prob = temporal_score
            source = "temporal-only (no faces detected)"

        if ai_prob >= 0.7:
            verdict = "likely AI-generated"
        elif ai_prob <= 0.3:
            verdict = "likely real"
        else:
            verdict = "ambiguous or mixed"

        cert = "high" if confidence >= 0.8 else ("moderate" if confidence >= 0.5 else "low")

        parts = []
        if face_present:
            parts.append(f"frame_score={frame_score:.2f}")
        parts.append(f"temporal_score={temporal_score:.2f}")
        parts.append(f"confidence={confidence:.2f}")

        reason = "; ".join(parts)
        return f"This video is {verdict} with {cert} confidence based on {source}: {reason}."
    except Exception:
        return "No explanation available."
