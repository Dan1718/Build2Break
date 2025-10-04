# app/detectors/sample_detector.py
import json
import os

def detect(input_path: str, meta: dict) -> dict:
    """
    Return:
    {
      "probability": float,  # 0..1
      "explanation": str,
      "raw": {...}
    }
    """
    # naive detection: if text contains the word "generated"
    try:
        # If text-like file:
        with open(input_path, "rb") as f:
            data = f.read(10000)
        try:
            text = data.decode("utf-8", errors="ignore").lower()
            if "generated" in text or "ai-generated" in text:
                prob = 0.85
                explanation = "found typical AI marker words in text"
            else:
                prob = 0.1
                explanation = "no textual markers found (toy detector)"
        except Exception:
            # Not text (audio/video). Very naive: check file size
            sz = os.path.getsize(input_path)
            if sz < 1000:
                prob = 0.05
                explanation = "tiny file"
            else:
                prob = 0.25
                explanation = "non-text file, heuristics low confidence"
        return {"probability": prob, "explanation": explanation, "raw": {"size": os.path.getsize(input_path)}}
    except Exception as e:
        return {"probability": 0.0, "explanation": f"detector error: {e}", "raw": {"error": str(e)}}

# CLI support (so FastAPI can run this as a subprocess too)
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    print(json.dumps(detect(args.input, {})))
