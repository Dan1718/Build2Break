# models/temporal_detector.py
import cv2
import numpy as np
import math

class TemporalDetector:
    """
    Temporal detector using OpenCV Farneback optical flow.
    Returns (temporal_score, magnitudes_list)
    temporal_score: [0,1], higher -> more anomalous
    magnitudes_list: mean flow magnitude per pair
    """

    def __init__(self, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0):
        self.fb_params = dict(
            pyr_scale=float(pyr_scale),
            levels=int(levels),
            winsize=int(winsize),
            iterations=int(iterations),
            poly_n=int(poly_n),
            poly_sigma=float(poly_sigma),
            flags=int(flags)
        )

    def detect(self, frames):
        try:
            if not frames or len(frames) < 2:
                return 0.0, []

            total = len(frames)
            step = max(1, total // 120)  # sample fewer pairs for performance
            sampled = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames[::step]]
            magnitudes = []

            for i in range(len(sampled) - 1):
                try:
                    prev = sampled[i]
                    nxt = sampled[i+1]
                    # ensure dtype uint8
                    prev_u = prev.astype(np.uint8)
                    nxt_u = nxt.astype(np.uint8)
                    flow = cv2.calcOpticalFlowFarneback(prev_u, nxt_u, None, **self.fb_params)
                    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
                    mean_mag = float(np.mean(mag))
                    magnitudes.append(mean_mag)
                except Exception:
                    continue

            if not magnitudes:
                return 0.0, []

            mags = np.array(magnitudes)
            mean_mag = float(mags.mean())
            std_mag = float(mags.std())

            rel_std = std_mag / (mean_mag + 1e-9)
            frozenness = math.exp(-mean_mag)  # higher when mean_mag is very small (frozen)
            raw = 0.6 * (rel_std / (1.0 + rel_std)) + 0.4 * frozenness
            score = float(np.clip(raw, 0.0, 1.0))
            return score, magnitudes
        except Exception:
            return 0.0, []
