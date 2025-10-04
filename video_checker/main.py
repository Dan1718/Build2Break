# main.py
import argparse
import json
import time
import traceback

from preprocessor import VideoPreprocessor
from detectors import VideoDetector

def main():
    parser = argparse.ArgumentParser(description="AI-Generated Video Detector (Python 3.13 compatible)")
    parser.add_argument("video_path", help="Path to input video")
    parser.add_argument("--resize_w", type=int, default=640, help="Frame resize width (default 640)")
    parser.add_argument("--resize_h", type=int, default=360, help="Frame resize height (default 360)")
    parser.add_argument("--max_frames", type=int, default=300, help="Max frames to extract (default 300)")
    parser.add_argument("--sample_fps", type=float, default=1.0, help="Approx FPS to sample (default 1.0)")
    args = parser.parse_args()

    start = time.time()
    output = {"status": "error", "details": "", "processing_time": None}

    try:
        pre = VideoPreprocessor(resize=(args.resize_w, args.resize_h),
                                max_frames=args.max_frames,
                                sample_fps=args.sample_fps)
        res, msg = pre.process(args.video_path)
        if res is None:
            output.update({
                "status": "error",
                "details": msg,
                "processing_time": round(time.time() - start, 3)
            })
            print(json.dumps(output))
            return

        (frames, metadata), _ = res, msg  # pre.process returns ((frames, metadata), "Success")
        detector = VideoDetector()
        result = detector.run_detection(frames)

        output.update({
            "status": "success",
            "ai_probability": result["ai_probability"],
            "confidence": result["confidence"],
            "frame_score": result["frame_score"],
            "temporal_score": result["temporal_score"],
            "face_present": result["face_present"],
            "explanation": result["explanation"],
            "metadata": metadata,
            "processing_time": round(time.time() - start, 3)
        })
    except Exception as e:
        output.update({
            "status": "error",
            "details": f"{str(e)}\n{traceback.format_exc()}",
            "processing_time": round(time.time() - start, 3)
        })

    # Print a single JSON object (always)
    print(json.dumps(output))

if __name__ == "__main__":
    main()
