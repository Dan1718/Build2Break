import csv
import torch
from transformers import pipeline

# ✅ Detect GPU; use CPU if not available
device = 0 if torch.cuda.is_available() else -1
classifier = pipeline(
    'zero-shot-classification',
    model='facebook/bart-large-mnli',
    device=device
)

def compute_confidence(prob, high=0.85, medium=0.6):
    """
    Converts probability to confidence label.
    """
    if prob >= high:
        return "High"
    elif prob >= medium:
        return "Medium"
    else:
        return "Low"

def predict_label(text):
    """
    Predicts label for text:
    Returns tuple: (pred_label, ai_probability, confidence)
    """
    labels = ['AI-generated', 'Human-written']
    result = classifier(text, candidate_labels=labels)
    
    ai_prob = result['scores'][result['labels'].index('AI-generated')]
    pred_label = round(ai_prob)  # 0 or 1
    confidence = compute_confidence(ai_prob)
    
    return pred_label, ai_prob, confidence

def evaluate_csv(csv_file_path):
    total = 0
    correct = 0

    with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for i, row in enumerate(reader, start=1):
            text = row['text']
            true_label = float(row['generated'])  # 0.0 = human, 1.0 = AI

            pred_label, ai_prob, confidence = predict_label(text)
            
            total += 1
            if pred_label == round(true_label):  # accuracy uses rounded labels
                correct += 1

            # Print progress every 10 rows
            if i % 10 == 0 or i == total:
                accuracy = correct / total
                print(f"Row {i}: True={true_label}, Predicted={pred_label}, "
                      f"Probability={ai_prob:.3f}, Confidence={confidence}, "
                      f"Accuracy so far={accuracy:.3f}")

    final_accuracy = correct / total if total > 0 else 0.0
    print(f"\nFinal Accuracy: {correct}/{total} = {final_accuracy:.3f}")

if __name__ == "__main__":
    csv_file_path = "./AI_Human.csv"  # Update path if needed
    evaluate_csv(csv_file_path)
