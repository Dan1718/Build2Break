# main.py
import argparse
import json
import time
import os

from preprocessor import VideoPreprocessor
from detectors import VideoDetector

def main():
    parser = argparse.ArgumentParser(description="AI-Generated Video Detector for Folder of Videos")
    parser.add_argument("folder_path", help="Path to folder containing video files")
    args = parser.parse_args()

    folder_path = args.folder_path
    if not os.path.isdir(folder_path):
        print(json.dumps({"status": "error", "details": "Folder not found"}))
        return

    video_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))]
    if not video_files:
        print(json.dumps({"status": "error", "details": "No video files found in folder"}))
        return

    results = []

    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        start = time.time()
        output = {"video_file": video_file, "status": "error", "details": "", "processing_time": None}

        try:
            pre = VideoPreprocessor()  # default parameters
            res, msg = pre.process(video_path)
            if res is None:
                output.update({
                    "status": "error",
                    "details": msg,
                    "processing_time": round(time.time() - start, 3)
                })
                results.append(output)
                continue

            frames, _ = res
            detector = VideoDetector()
            result = detector.run_detection(frames)

            output.update({
                "status": "success",
                "ai_probability": result["ai_probability"],
                "confidence": result["confidence"],
                "explanation": result.get("explanation", ""),
                "processing_time": round(time.time() - start, 3)
            })
        except Exception as e:
            output.update({
                "status": "error",
                "details": str(e),
                "processing_time": round(time.time() - start, 3)
            })

        results.append(output)

    # Print results as JSON array
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
