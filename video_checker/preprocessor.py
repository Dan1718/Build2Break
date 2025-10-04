# preprocessor.py
import os
import cv2
import mimetypes

# try to import python-magic; if missing, we'll fall back to mimetypes
try:
    import magic as _magic  # python-magic
except Exception:
    _magic = None

class VideoPreprocessor:
    """
    Robust video preprocessor:
      - validates file existence, readability, MIME type (magic or mimetypes)
      - extracts sampled frames (resized) with defensive error handling
      - returns (frames_list, metadata) on success, or (None, error_message) on failure
    """

    def __init__(self, resize=(640, 360), max_frames=300, sample_fps=1.0):
        """
        resize: target (width, height) for frames (keeps processing fast)
        max_frames: cap on number of extracted frames
        sample_fps: approximate FPS to sample (if 0 => uniform sampling up to max_frames)
        """
        self.resize = resize
        self.max_frames = max(1, int(max_frames))
        self.sample_fps = float(sample_fps) if sample_fps is not None else 1.0

    def _looks_like_video(self, path):
        try:
            if _magic is not None:
                m = _magic.from_file(path, mime=True)
                return isinstance(m, str) and m.startswith("video")
            else:
                t, _ = mimetypes.guess_type(path)
                return t is not None and t.startswith("video")
        except Exception:
            return False

    def process(self, video_path):
        # Basic validation
        try:
            if not isinstance(video_path, str):
                return None, "video_path must be a string"

            if not os.path.exists(video_path):
                return None, f"File does not exist: {video_path}"

            if not os.access(video_path, os.R_OK):
                return None, f"File is not readable: {video_path}"

            if os.path.getsize(video_path) == 0:
                return None, f"File is empty: {video_path}"

            if not self._looks_like_video(video_path):
                # be permissive for common extensions
                ext = os.path.splitext(video_path)[1].lower()
                if ext not in (".mp4", ".mov", ".mkv", ".avi", ".webm"):
                    return None, f"File does not appear to be a video (MIME check failed): {video_path}"
        except Exception as e:
            return None, f"File validation error: {e}"

        # Open capture
        cap = None
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None, f"Failed to open video with OpenCV: {video_path}"

            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            duration = frame_count / (fps if fps > 0 else 30.0)

            # Choose indices to sample
            indices = []
            if frame_count <= 0:
                # unknown length - read until EOF but cap frames
                # We'll just read sequentially until max_frames or EOF
                idx = 0
                while len(indices) < self.max_frames:
                    indices.append(idx)
                    idx += 1
            else:
                if self.sample_fps and self.sample_fps > 0:
                    step = max(1, int(round(fps / max(0.0001, self.sample_fps))))
                    for i in range(0, frame_count, step):
                        indices.append(i)
                        if len(indices) >= self.max_frames:
                            break
                else:
                    # uniform sampling up to max_frames
                    if frame_count <= self.max_frames:
                        indices = list(range(frame_count))
                    else:
                        stepf = frame_count / float(self.max_frames)
                        indices = [int(i * stepf) for i in range(self.max_frames)]

            # Extract frames defensively
            frames = []
            last_idx = -1
            for idx in indices:
                try:
                    if idx == last_idx:
                        continue
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    last_idx = idx
                    if not ret or frame is None:
                        continue
                    if self.resize:
                        try:
                            frame = cv2.resize(frame, self.resize, interpolation=cv2.INTER_AREA)
                        except Exception:
                            # If resize fails, keep original
                            pass
                    frames.append(frame)
                    if len(frames) >= self.max_frames:
                        break
                except Exception:
                    # skip problematic frames silently
                    continue

            cap.release()

            if not frames:
                return None, "No frames extracted (video may be corrupted or unreadable)"

            metadata = {
                "fps": float(fps),
                "frame_count": int(frame_count),
                "duration_sec": float(duration),
                "frames_extracted": int(len(frames))
            }
            return (frames, metadata), "Success"

        except Exception as e:
            try:
                if cap:
                    cap.release()
            except Exception:
                pass
            return None, f"Error processing video: {e}"
