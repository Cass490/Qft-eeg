from flask import Flask, render_template, jsonify, request
import os
import numpy as np
import pandas as pd
from inference import DemoInference
import torch
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '../qvae_eeg_model.pth')
DATA_PATH = os.path.join(BASE_DIR, '../data/emotions.csv')

from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '../qvae_eeg_model.pth')
DATA_PATH = os.path.join(BASE_DIR, '../data/emotions.csv')

# 1. Initialize Objects
print("Fitting scaler on emotions.csv...")
df_full = pd.read_csv(DATA_PATH)
labels_raw = df_full['label'].values
features_raw = df_full.drop('label', axis=1).values

# Scaler
scaler = StandardScaler()
X_norm = scaler.fit_transform(features_raw)

# Inference Model
infer = DemoInference(MODEL_PATH)

# 2. Train Internal Classifier (Match the Architecture Document claim)
print("Generating Latent Space for Internal Classifier Training...")
with torch.no_grad():
    # Pass data in chunks to avoid OOM or quantum simulation bottleneck
    # For speed in demo startup, we'll use a subset (e.g., 500 samples)
    subset_size = min(500, len(X_norm))
    X_subset = torch.FloatTensor(X_norm[:subset_size])
    _, train_latents = infer.model(X_subset)
    y_subset = labels_raw[:subset_size]

print("Training Logistic Regression on Quantum Latent features...")
clf = LogisticRegression(max_iter=1000)
clf.fit(train_latents.numpy(), y_subset)

del df_full # Save memory

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/sample')
def get_sample():
    """Get a random sample from the emotions.csv to test"""
    if not os.path.exists(DATA_PATH):
        # Fallback to random data if csv not found
        sample = np.random.rand(2548).tolist()
        return jsonify({"data": sample, "label": "Random (Sample not found)"})
    
    df = pd.read_csv(DATA_PATH).sample(1)
    label = df['label'].values[0]
    data = df.drop('label', axis=1).values[0].tolist()
    
    return jsonify({"data": data, "label": label})

@app.route('/api/predict', methods=['POST'])
def predict():
    content = request.json
    raw_data = np.array(content['data']).reshape(1, -1)
    
    # 1. Normalize (as expected by model)
    normalized_data = scaler.transform(raw_data)
    
    # 2. Run Inference
    results = infer.run(normalized_data)
    
    # Get outputs (stays in normalized domain for best visualization match)
    reconstructed_norm = results['reconstructed'][0]
    latent = results['latent']
    
    # 3. Use the trained classifier for real accuracy
    emotion = clf.predict(latent)[0]
    emotion_probs = clf.predict_proba(latent)[0]
    conf = float(np.max(emotion_probs)) * 100
    
    return jsonify({
        "input_norm": normalized_data[0].tolist(),
        "reconstructed": reconstructed_norm.tolist(),
        "latent": latent[0].tolist(),
        "emotion": str(emotion),
        "confidence": f"{conf:.2f}%"
    })

if __name__ == '__main__':
    print("Starting QVAE Demo Server at http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
