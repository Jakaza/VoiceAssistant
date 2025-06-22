import sounddevice as sd
from scipy.io.wavfile import write
from features import extract_features
from joblib import load
import numpy as np
import os

SAMPLE_RATE = 16000
DURATION = 2  # seconds
MODEL_PATH = "model.joblib"

def record_temp_command():
    print("Speak your command...")
    audio = sd.rec(int(SAMPLE_RATE*DURATION), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()

    filename = "temp.wav"
    write(filename, SAMPLE_RATE, audio)

    return filename

def predict_command():
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Please train the model first.")
        return
    
    clf, le = load(MODEL_PATH)
    filename = record_temp_command()
    features = extract_features(filename).reshape(1 , -1)
    prediction = clf.predict(features)[0]
    command = le.inverse_transform([prediction])[0]

    print(f"You said: {command}")
    return command


predict_command()
