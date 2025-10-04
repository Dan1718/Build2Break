# models/frame_detector.py
import cv2
import numpy as np
import math

class FrameDetector:
    """
    Heuristic frame detector (no ML):
      - face presence (Haar cascade)
      - sharpness (Laplacian variance)
      - color anomaly (HSV skin-like fraction)
      - blockiness / compression artifacts (FFT-based proxy)
    Returns (score, face_present) where score in [0.0,1.0]
    """

    def __init__(self, face_cascade_path=None):
        if face_cascade_path is None:
            face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        try:
            self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
            if self.face_cascade.empty():
                self.face_cascade = None
        except Exception:
            self.face_cascade = None

    def _detect_faces(self, frame):
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self.face_cascade is None:
                return []
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30,30))
            return list(faces)
        except Exception:
            return []

    def _sharpness(self, gray):
        try:
            v = cv2.Laplacian(gray, cv2.CV_64F).var()
            # Map variance to [0,1] with a smooth transform
            s = 1.0 - 1.0 / (1.0 + math.log1p(v + 1e-9))
            return float(np.clip(s, 0.0, 1.0))
        except Exception:
            return 0.5

    def _color_anomaly(self, frame):
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            mask = ((h > 0) & (h < 25)) & (s > 20) & (v > 40)
            skin_frac = float(np.count_nonzero(mask)) / (frame.shape[0] * frame.shape[1] + 1e-9)
            target = 0.1
            diff = abs(skin_frac - target)
            score = diff / (0.5 + diff)
            return float(np.clip(score, 0.0, 1.0))
        except Exception:
            return 0.5

    def _blockiness(self, gray):
        try:
            f = np.fft.fft2(gray.astype(np.float32))
            fshift = np.fft.fftshift(f)
            mag = np.abs(fshift)
            h, w = mag.shape
            center_r = int(min(h, w) * 0.08) + 1
            cy, cx = h//2, w//2
            yy, xx = np.ogrid[:h, :w]
            mask_center = (yy - cy)**2 + (xx - cx)**2 <= center_r**2
            low = mag[mask_center].sum()
            high = mag[~mask_center].sum() + 1e-9
            ratio = low / high
            score = 1.0 - (1.0 / (1.0 + ratio))
            return float(np.clip(score, 0.0, 1.0))
        except Exception:
            return 0.5

    def detect(self, frames, face_sensitive=True):
        """
        frames: list of BGR images (numpy arrays)
        face_sensitive: if True, detection will look for faces and set face_present accordingly
        returns (score, face_present)
        """
        try:
            if not frames:
                return 0.0, False

            total = len(frames)
            step = max(1, total // 30)  # sample up to ~30 frames
            sample = frames[::step][:30]

            scores = []
            face_present = False
            for f in sample:
                try:
                    gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                except Exception:
                    continue

                faces = self._detect_faces(f) if face_sensitive else []
                if faces:
                    face_present = True

                sharp = self._sharpness(gray)
                color = self._color_anomaly(f)
                blocky = self._blockiness(gray)

                # Conservative weighted sum tuned to avoid false positives
                s = 0.35 * blocky + 0.35 * color + 0.30 * sharp
                scores.append(s)

            if not scores:
                return 0.0, face_present

            mean_score = float(np.mean(scores))
            final = float(np.clip(mean_score, 0.0, 1.0))
            return final, face_present
        except Exception:
            return 0.0, False
