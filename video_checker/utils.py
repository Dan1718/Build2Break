# utils.py
import numpy as np

def calculate_confidence(frame_score, temporal_score, magnitudes=None):
    """
    Computes confidence based on agreement between frame and temporal detectors,
    optionally factoring in temporal consistency (entropy of motion magnitudes).
    Returns float [0,1].
    """
    base_conf = 1.0 - abs(frame_score - temporal_score)

    if magnitudes and len(magnitudes) > 1:
        # Temporal consistency: higher entropy (variability) lowers confidence
        entropy = np.std(magnitudes) / (np.mean(magnitudes) + 1e-6)  # normalized std
        base_conf *= np.clip(1.0 - entropy, 0.0, 1.0)

    return float(np.clip(base_conf, 0.0, 1.0))
