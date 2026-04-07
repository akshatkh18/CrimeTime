import pandas as pd
import numpy as np
import pickle
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

CLEANED_PATH = os.path.join(os.path.dirname(__file__), 'data', 'cleaned.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'rf_model.pkl')
ENCODER_PATH = os.path.join(os.path.dirname(__file__), 'model', 'encoders.pkl')
METRICS_PATH = os.path.join(os.path.dirname(__file__), 'model', 'metrics.json')

FEATURE_COLS = [
    'MURDER', 'ATTEMPT TO MURDER',
    'CULPABLE HOMICIDE NOT AMOUNTING TO MURDER',
    'RAPE', 'KIDNAPPING & ABDUCTION',
    'DACOITY', 'ROBBERY', 'BURGLARY', 'THEFT',
    'RIOTS', 'CRIMINAL BREACH OF TRUST', 'CHEATING',
    'ARSON', 'HURT/GREVIOUS HURT', 'DOWRY DEATHS',
    'ASSAULT ON WOMEN WITH INTENT TO OUTRAGE HER MODESTY',
    'CRUELTY BY HUSBAND OR HIS RELATIVES',
    'CAUSING DEATH BY NEGLIGENCE', 'OTHER IPC CRIMES',
    'TOTAL IPC CRIMES', 'YEAR'
]

def train():
    df = pd.read_csv(CLEANED_PATH)

    # Encode state and district
    le_state = LabelEncoder()
    le_district = LabelEncoder()
    df['STATE_ENC'] = le_state.fit_transform(df['STATE/UT'])
    df['DISTRICT_ENC'] = le_district.fit_transform(df['DISTRICT'])

    features = FEATURE_COLS + ['STATE_ENC', 'DISTRICT_ENC']
    X = df[features]
    y = df['SAFETY_LABEL']

    le_label = LabelEncoder()
    y_enc = le_label.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le_label.classes_, output_dict=True)

    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=le_label.classes_))

    # Save model and encoders
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)

    with open(ENCODER_PATH, 'wb') as f:
        pickle.dump({
            'state': le_state,
            'district': le_district,
            'label': le_label,
            'features': features
        }, f)

    metrics = {
        "accuracy": round(acc, 4),
        "report": report,
        "feature_importance": dict(zip(features, model.feature_importances_.round(4).tolist()))
    }
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)

    print("Model and encoders saved.")
    return model

if __name__ == '__main__':
    train()
