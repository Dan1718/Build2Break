import io
import librosa
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import soundfile as sf

def compute_confidence(p1, p2, threshold_high=0.85, threshold_medium=0.65):
    mass_score = (p1 + p2)
    
    separation_score = abs(p1 - p2)
    
    combined_score = mass_score * separation_score
    
    if combined_score >= threshold_high:
        return "High"
    elif combined_score >= threshold_medium:
        return "Medium"
    else:
        return "Low"

def extract_mfcc_features_from_bytes(file_bytes, n_mfcc=13, n_fft=2048, hop_length=512):
    """
    Extract MFCC features from an audio file provided as bytes.
    """
    try:
        # Load from bytes
        audio_data, sr = sf.read(io.BytesIO(file_bytes), dtype="float32")

        # If stereo, convert to mono
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)

        mfccs = librosa.feature.mfcc(
            y=audio_data, sr=sr,
            n_mfcc=n_mfcc, n_fft=n_fft,
            hop_length=hop_length
        )
        return np.mean(mfccs.T, axis=0)

    except Exception as e:
        print(f"Error extracting features: {e}")
        return None


def analyze_audio_bytes(file_bytes, model_path="svm_model.pkl", scaler_path="scaler.pkl"):
    """
    Analyze audio given as raw bytes.
    Returns classification result string.
    """
    # Extract MFCC features
    mfcc_features = extract_mfcc_features_from_bytes(file_bytes)
    if mfcc_features is None:
        return "Error: Unable to process the input audio."

    # Load scaler + transform features
    try:
        scaler = joblib.load(scaler_path)
        mfcc_features_scaled = scaler.transform(mfcc_features.reshape(1, -1))
    except Exception as e:
        return f"Error loading scaler: {e}"

    # Load SVM model and predict
    try:
        svm_classifier = joblib.load(model_path)
        probabilities = svm_classifier.predict_proba(mfcc_features_scaled)[0]

        genuine_prob, deepfake_prob = probabilities[0], probabilities[1]
        p1, p2 = sorted([genuine_prob, deepfake_prob], reverse=True)

        
        return (round(deepfake_prob,2),compute_confidence(deepfake_prob,genuine_prob),'navyblueeueue')
        
    except Exception as e:
        return f"Error loading model or predicting: {e}"

if __name__ == '__main__':

    with open("./deepfake_audio/file12.wav", "rb") as f:
        audio_bytes = f.read()

    result = analyze_audio_bytes(audio_bytes)
    print(result)
