# preprocessor.py
import os
import cv2
import magic
import numpy as np

class VideoPreprocessor:
    def __init__(self, resize=(224, 224)):
        self.resize = resize

    def process(self, video_path):
        if not os.path.exists(video_path):
            return None, f"File {video_path} does not exist."
        if not os.access(video_path, os.R_OK):
            return None, f"File {video_path} is not readable."
        if os.path.getsize(video_path) == 0:
            return None, f"File {video_path} is empty."

        mime = magic.from_file(video_path, mime=True)
        if not mime.startswith("video"):
            return None, f"File {video_path} is not a video (MIME={mime})."

        frames = []
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None, f"Failed to open video {video_path}."

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if self.resize:
                    frame = cv2.resize(frame, self.resize)
                frames.append(frame)
            cap.release()

            if len(frames) == 0:
                return None, "No frames extracted from video."

        except Exception as e:
            return None, f"Error processing video: {str(e)}"

        return frames, "Success"
