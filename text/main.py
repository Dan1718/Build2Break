# text_detector_app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import pipeline


device = 0 if torch.cuda.is_available() else -1
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=device
)

 
app = FastAPI(title="Text AI Detector")


class TextInput(BaseModel):
    text: str

class TextAnalysisResponse(BaseModel):
    explanation: str
    probability: float
    confidence: str


def compute_confidence(prob, high=0.85, medium=0.6):
    if prob >= high:
        return "High"
    elif prob >= medium:
        return "Medium"
    else:
        return "Low"


@app.post("/analyze_text", response_model=TextAnalysisResponse)
def analyze_text(input: TextInput):
    if not input.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    labels = ["AI-generated", "Human-written"]
    result = classifier(input.text, candidate_labels=labels)

    ai_prob = result["scores"][result["labels"].index("AI-generated")]
    predicted_label = "AI-generated" if ai_prob >= 0.5 else "Human-written"
    confidence = compute_confidence(ai_prob)
    print(ai_prob)
    return TextAnalysisResponse(
        explanation="explanation",
        probability=ai_prob,
        confidence=confidence
    )

@app.get("/")
def root():
    return {"message": "Text AI Detector is running. POST text to /analyze_text"}
