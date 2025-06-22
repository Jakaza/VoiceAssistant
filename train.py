import os
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump
from features import extract_features

AUDIO_DIR = "audio/"
MODEL_PATH = "model.joblib"

def load_data():
    x = []
    y = []

    for label in os.listdir(AUDIO_DIR) :
        folder = os.path.join(AUDIO_DIR , label)
        if not os.path.isdir(folder): continue

        for file in os.listdir(folder):
            if file.endswith(".wav"):
                file_path = os.path.join(folder, file)
                features = extract_features(file_path)
                x.append(features)
                y.append(label)

    return np.array(x), np.array(y)


