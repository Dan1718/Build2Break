import csv
from transformers import pipeline

classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')

def corrected_human_score(human_raw):
    if human_raw < 0.65:
        return 0.0
    elif human_raw < 0.8:
        return 0.03 + (human_raw - 0.65) * (0.1 - 0.03) / (0.8 - 0.65)
    elif human_raw < 0.9:
        return 0.1 + (human_raw - 0.8) * (1.0 - 0.1) / (0.9 - 0.8)
    else:
        return 1.0

def predict_label(text):
    labels = ['AI-generated', 'Human-written']
    result = classifier(text, candidate_labels=labels)
    human_raw = result['scores'][result['labels'].index('Human-written')]
    human_score = corrected_human_score(human_raw)
    
    if human_score >= 0.5:
        return "human"
    elif human_score >= 0.3:
        return "human" if human_score >= 0.4 else "ai"  # deterministic split at 0.4
    else:
        return "ai"

csv_file_path = "../AI_Human.csv"  # change to your CSV path
total = 0
correct = 0

with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for i, row in enumerate(reader, start=1):
        text = row['text']
        true_label = row['label'].lower()  # "human" or "ai"
        pred_label = predict_label(text)

        total += 1
        if pred_label == true_label:
            correct += 1

        accuracy = correct / total
        print(f"Row {i}: True={true_label}, Predicted={pred_label}, Accuracy so far={accuracy:.3f}")

print(f"\nFinal Accuracy: {correct}/{total} = {correct/total:.3f}")
