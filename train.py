import os
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump
from features import extract_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

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

def train_model():
    print("Training voice command model...")

    x, y = load_data()

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(x, y_encoded,test_size=0.2)

    clf = RandomForestClassifier(n_estimators=100)
    # clf = SVC(kernel="linear", probability=True )
    clf.fit(X_train,y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Model trained with accuracy: {acc:.2f}")

    print(classification_report(y_test, y_pred, target_names=le.classes_))

    dump((clf, le), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

train_model()