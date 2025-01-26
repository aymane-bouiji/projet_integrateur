# maj_modele.py
from flask import Flask, jsonify
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from joblib import dump, load
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
import logging

# Initialize Flask
app = Flask(__name__)

# Paths
LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))
PV_DIR = "/shared_volume/"

# NLTK Setup
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def Clean_message(text):
    """Text preprocessing identical in both services"""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

@app.route('/train', methods=['POST'])
def train():
    """Train model and save to PV"""
    try:
        # Load existing models or initialize new ones
        try:
            vectorizer = load(os.path.join(LOCAL_DIR, 'vectorizer.joblib'))
            model = load(os.path.join(LOCAL_DIR, 'random_forest_model.joblib'))
            label_encoder = load(os.path.join(LOCAL_DIR, 'label_encoder.joblib'))
        except FileNotFoundError:
            vectorizer = CountVectorizer()
            model = RandomForestClassifier()
            label_encoder = LabelEncoder()

        # Load and preprocess data
        dataset = pd.read_csv(os.path.join(LOCAL_DIR, 'messages.csv'))
        dataset['MainText'] = dataset['MainText'].apply(Clean_message)
        dataset['label'] = dataset['label'].replace(
            {'smishing': 'phishing', 'spam': 'phishing', 'malware': 'phishing'}
        )
        dataset = dataset[dataset['label'].isin(['ham', 'phishing'])].sample(frac=1, random_state=42)

        # Train
        X = vectorizer.fit_transform(dataset['MainText'])
        y = label_encoder.fit_transform(dataset['label'])
        model.fit(X, y)

        # Save to PV
        dump(vectorizer, os.path.join(PV_DIR, 'vectorizer.joblib'))
        dump(model, os.path.join(PV_DIR, 'random_forest_model.joblib'))
        dump(label_encoder, os.path.join(PV_DIR, 'label_encoder.joblib'))

        return jsonify({
            "status": "success",
            "message": "Model retrained and saved to PV",
            "dataset_size": len(dataset)
        }), 200

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5004)
