# maj_server.py
from flask import Flask, jsonify, request
from joblib import load
import numpy as np
import os
import logging

# Initialize Flask
app = Flask(__name__)

# Paths
PV_DIR = "/shared_volume/"

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load models from PV on startup
try:
    vectorizer = load(os.path.join(PV_DIR, 'vectorizer.joblib'))
    model = load(os.path.join(PV_DIR, 'random_forest_model.joblib'))
    label_encoder = load(os.path.join(PV_DIR, 'label_encoder.joblib'))
    logger.info("Models loaded successfully from PV")
except FileNotFoundError:
    logger.error("Models not found in PV! Deploy training server first.")
    raise

def Clean_message(text):
    """Identical preprocessing to training service"""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions using PV models"""
    try:
        data = request.json
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({"error": "Empty input"}), 400

        # Preprocess and predict
        cleaned_text = Clean_message(text)
        X = vectorizer.transform([cleaned_text])
        prediction = model.predict(X)
        label = label_encoder.inverse_transform(prediction)[0]

        return jsonify({
            "prediction": label,
            "processed_text": cleaned_text
        }), 200

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Kubernetes liveness probe"""
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
