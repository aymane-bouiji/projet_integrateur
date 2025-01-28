from flask import Flask, request, jsonify
import pandas as pd
import os
import requests
from joblib import load
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Configuration
LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_CSV = os.path.join(LOCAL_DIR, 'messages.csv')
TRAINING_THRESHOLD = 2

# NLTK Initialization
nltk.download('stopwords')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_message(text):
    """Preprocess text identically to training server"""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = ' '.join(word for word in text.split() if word not in stop_words)
    text = ' '.join(lemmatizer.lemmatize(word) for word in text.split())
    return text

@app.route('/add_message', methods=['POST'])
def add_message():
    data = request.json
    messages = data.get('messages', [])
    labels = data.get('labels', [])
    
    if len(messages) != len(labels):
        return jsonify({"error": "Messages and labels mismatch!"}), 400

    # Load existing data
    df = pd.DataFrame(columns=['MainText', 'label'])
    if os.path.exists(LOCAL_CSV):
        df = pd.read_csv(LOCAL_CSV)

    # Add new data
    new_df = pd.DataFrame({'MainText': messages, 'label': labels})
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(LOCAL_CSV, index=False)

    # Trigger training if threshold met
    if len(df) >= TRAINING_THRESHOLD:
        try:
            response = requests.post("http://localhost:5001/train")
            if response.status_code == 200:
                # Reset CSV for next batch
                pd.DataFrame(columns=['MainText', 'label']).to_csv(LOCAL_CSV, index=False)
                return jsonify({
                    "message": f"Trained on {len(df)} messages.",
                    "new_samples_needed": TRAINING_THRESHOLD
                })
            else:
                return jsonify({"error": "Training failed"}), 500
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({
            "message": f"Added {len(messages)} messages. {TRAINING_THRESHOLD - len(df)} more needed.",
            "current_total": len(df)
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5009)
