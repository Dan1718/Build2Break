# models/temporal_detector.py
import os
import sys
import torch
import numpy as np

# Add RAFT repo to path (adjusted for your nested clone)
sys.path.append(os.path.join(os.path.dirname(video_checker/raft/RAFT), "../raft/RAFT"))

try:
    from core.raft import RAFT
    from core.utils.utils import InputPadder
except ImportError as e:
    raise ImportError(
        "Could not import RAFT. Make sure your RAFT clone is in 'video_checker/raft/RAFT'.\n"
        f"Original error: {e}"
    )

class TemporalDetector:
    """
    Motion-based AI detector using RAFT optical flow.
    Returns a normalized temporal anomaly score.
    """
    def __init__(self, weights_path="models/raft-sintel.pth", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"RAFT weights not found: {weights_path}")

        # Load RAFT model
        self.model = RAFT()
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def detect(self, frames):
        """
        Computes temporal anomaly score from a list of frames.
        Returns:
            score: float [0.0, 1.0]
            magnitudes: list of float magnitudes for each frame pair
        """
        if len(frames) < 2:
            return 0.0, []

        magnitudes = []

        for i in range(len(frames)-1):
            try:
                im1 = torch.from_numpy(frames[i].transpose(2,0,1)).unsqueeze(0).float().to(self.device) / 255.0
                im2 = torch.from_numpy(frames[i+1].transpose(2,0,1)).unsqueeze(0).float().to(self.device) / 255.0
                padder = InputPadder(im1.shape)
                im1, im2 = padder.pad(im1, im2)

                flow_low, flow_up = self.model(im1, im2, iters=20, test_mode=True)
                mag = torch.mean(torch.norm(flow_up[0], dim=0)).item()
                magnitudes.append(mag)

            except Exception as e:
                print(f"[TemporalDetector] Skipped frame pair {i}-{i+1} due to error: {e}")
                continue

        if not magnitudes:
            return 0.0, []

        # Smooth temporal anomaly score: higher motion = lower anomaly
        mean_mag = np.mean(magnitudes)
        score = float(np.clip(1 - np.exp(-mean_mag), 0.0, 1.0))

        return score, magnitudes
