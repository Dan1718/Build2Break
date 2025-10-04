# app/utils.py
import os, tempfile, shutil, json, sys, subprocess, importlib
from typing import List, Dict, Optional

DETECTORS_PACKAGE = "app.detectors"
DETECTORS_DIR = os.path.join(os.path.dirname(__file__), "detectors")
DEFAULT_TIMEOUT = 50  # seconds

def save_upload_to_tempfile(upload_file) -> str:
    suffix = ""
    if upload_file.filename:
        _, ext = os.path.splitext(upload_file.filename)
        suffix = ext
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="upload_")
    os.close(fd)
    with open(path, "wb") as f:
        contents = upload_file.file.read()
        f.write(contents)
    return path

def list_detectors() -> List[str]:
    # detectors as python modules or detector_name.py file
    names = []
    if os.path.isdir(DETECTORS_DIR):
        for fname in os.listdir(DETECTORS_DIR):
            if fname.startswith("__"): 
                continue
            if fname.endswith(".py"):
                names.append(fname[:-3])
            elif os.path.isdir(os.path.join(DETECTORS_DIR, fname)):
                names.append(fname)
    return names

def run_detector(input_path: str, detectors_csv: Optional[str] = None, timeout:int=DEFAULT_TIMEOUT) -> List[Dict]:
    """
    Runs detectors and returns list of result dicts:
    [{
        "detector": "sample_detector",
        "probability": 0.7,
        "explanation": "...",
        "raw": {...}
    }, ...]
    """
    if detectors_csv:
        detectors = [d.strip() for d in detectors_csv.split(",") if d.strip()]
    else:
        detectors = list_detectors()

    results = []
    for det in detectors:
        try:
            # Try importing Python module
            mod_name = f"{DETECTORS_PACKAGE}.{det}"
            try:
                mod = importlib.import_module(mod_name)
                if hasattr(mod, "detect"):
                    # call directly (fast, in-process)
                    out = mod.detect(input_path, meta={})
                    out = normalize_detector_output(det, out)
                else:
                    raise ImportError("no detect() in module")
            except Exception:
                # fallback to CLI: look for detectors/<det>.py or detectors/<det>/run
                cli_path = os.path.join(DETECTORS_DIR, f"{det}.py")
                if os.path.exists(cli_path):
                    cmd = [sys.executable, cli_path, "--input", input_path]
                else:
                    # try detectors/det/run_detector.sh or entrypoint
                    alt = os.path.join(DETECTORS_DIR, det, "run_detector.sh")
                    if os.path.exists(alt):
                        cmd = [alt, input_path]
                    else:
                        raise FileNotFoundError(f"Detector {det} not found as module or CLI")
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
                if proc.returncode != 0:
                    raise RuntimeError(f"CLI detector {det} failed: {proc.stderr[:400]}")
                out = json.loads(proc.stdout)
                out = normalize_detector_output(det, out)
        except Exception as e:
            # error -> produce a fallback result
            out = {
                "detector": det,
                "probability": 0.0,
                "explanation": f"error running detector: {e}",
                "raw": {"error": str(e)}
            }
        results.append(out)
    return results

def normalize_detector_output(detector_name: str, out: dict) -> dict:
    # Standardize keys and types
    if not isinstance(out, dict):
        raise ValueError("detector output must be a dict")
    prob = float(out.get("probability", out.get("score", 0.0)))
    explanation = out.get("explanation", out.get("explain", ""))
    raw = out.get("raw", out)
    return {"detector": detector_name, "probability": prob, "explanation": explanation, "raw": raw}

def aggregate_results(probs: List[float]) -> float:
    if not probs:
        return 0.0
    # simple average, you can swap to weighted voting
    return float(sum(probs)) / len(probs)

def confidence_from_prob(p: float) -> str:
    if p < 0.33:
        return "Low"
    if p < 0.66:
        return "Medium"
    return "High"
