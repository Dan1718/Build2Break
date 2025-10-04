# main.py
import argparse
import json
import time
import traceback
from preprocessor import VideoPreprocessor
from detectors import VideoDetector

def main():
    parser = argparse.ArgumentParser(description="AI-Generated Video Detector")
    parser.add_argument("video_path", type=str, help="Path to the input video")
    parser.add_argument("--frame_model", type=str, default="models/xception_weights.pth", help="Frame detector weights")
    parser.add_argument("--raft_model", type=str, default="models/raft-sintel.pth", help="RAFT weights")
    args = parser.parse_args()

    start_time = time.time()

    try:
        # Step 1: Preprocess video
        preprocessor = VideoPreprocessor()
        frames, status = preprocessor.process(args.video_path)

        if frames is None:
            # Failed preprocessing
            result = {
                "status": "error",
                "details": status,
                "processing_time": round(time.time() - start_time, 2)
            }
        else:
            # Step 2: Run detection
            detector = VideoDetector(frame_model_path=args.frame_model, raft_weights_path=args.raft_model)
            detection_result = detector.run_detection(frames)

            # Step 3: Prepare output
            result = {
                "status": "success",
                "ai_probability": detection_result["ai_probability"],
                "confidence": detection_result["confidence"],
                "frame_score": detection_result["frame_score"],
                "temporal_score": detection_result["temporal_score"],
                "explanation": detection_result["explanation"],
                "processing_time": round(time.time() - start_time, 2)
            }

    except Exception as e:
        # Catch all errors to prevent crash
        result = {
            "status": "error",
            "details": f"{str(e)}\n{traceback.format_exc()}",
            "processing_time": round(time.time() - start_time, 2)
        }

    # Step 4: Print single JSON object
    print(json.dumps(result, indent=4))

if __name__ == "__main__":
    main()
