from transformers import pipeline
import math

# Initialize zero-shot classifier
classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')

# Correction function for human-likeness (tempered mapping)
def corrected_human_score(human_raw):
    if human_raw < 0.65:
        return 0.0
    elif human_raw < 0.8:
        return 0.03 + (human_raw - 0.65) * (0.1 - 0.03) / (0.8 - 0.65)
    elif human_raw < 0.9:
        return 0.1 + (human_raw - 0.8) * (1.0 - 0.1) / (0.9 - 0.8)
    else:
        return 1.0

# Entropy calculator (normalized 0-1)
def calc_entropy(probs):
    return -sum(p * math.log(p + 1e-12) for p in probs) / math.log(2)

def detect_likeness_with_entropy_conf(text):
    labels = ['AI-generated', 'Human-written']
    result = classifier(text, candidate_labels=labels)

    # Extract raw scores
    human_raw = result['scores'][result['labels'].index('Human-written')]
    ai_raw = result['scores'][result['labels'].index('AI-generated')]

    # Correct human score
    human_likeness = corrected_human_score(human_raw)
    ai_likeness = 1.0 - human_likeness

    # Assessment thresholds
    if human_likeness > 0.5:
        assessment = "Likely Human"
    elif human_likeness >= 0.3:
        assessment = "Uncertain"
    else:
        assessment = "Likely AI"

    # Calculate entropy
    probs = [ai_likeness, human_likeness]
    entropy = calc_entropy(probs)

    # Adjust confidence based on class and entropy
    if assessment == "Likely Human":
        confidence_pct = round((1.0 - entropy * 0.7) * 100, 1)  # high if human even with some entropy
        reason = "High human-likeness with relatively low uncertainty."
    elif assessment == "Likely AI":
        confidence_pct = round((1.0 - entropy) * 100, 1)        # lower if AI but uncertain
        reason = "Low human-likeness and AI-likeness dominant with moderate uncertainty."
    else:
        confidence_pct = round((1.0 - entropy * 1.2) * 100, 1)  # uncertain → lower confidence
        reason = "Human-likeness and AI-likeness are similar, high uncertainty."

    return {
        "ai_likeness": round(ai_likeness, 3),
        "assessment": assessment,
        "confidence_pct": confidence_pct,
        "reason": reason
    }

if __name__ == "__main__":
    text = input("Enter text to analyze: ")
    result = detect_likeness_with_entropy_conf(text)

    print(f"AI-likeness: {result['ai_likeness']}")
    print(f"Assessment: {result['assessment']}")
    print(f"Confidence: {result['confidence_pct']}%")
    print(f"Reason: {result['reason']}")
