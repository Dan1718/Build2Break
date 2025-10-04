# Normalization anchors
RAW_MIN = 0.65
RAW_MAX = 0.9
NORM_MIN = 0.0
NORM_MAX = 1.0

def normalize_human_score(raw_human):
    """
    Input: raw human score (0-1)
    Output: normalized human-likeness (0-1)
    """
    # Linear mapping
    norm = (raw_human - RAW_MIN) / (RAW_MAX - RAW_MIN) * (NORM_MAX - NORM_MIN) + NORM_MIN
    # Clamp to 0–1
    return max(0.0, min(1.0, norm))

if __name__ == "__main__":
    raw_input_score = float(input("Enter raw human score (0-1): "))
    normalized = normalize_human_score(raw_input_score)
    print(f"Normalized Human-likeness: {normalized:.3f}")
