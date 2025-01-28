import os
from flask import Flask, request, jsonify
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the Flask application
app = Flask(__name__)

# Define the path to the volume
VOLUME_PATH = "/shared_volume"

# Define the paths to the model files in the volume
MODEL_PATH = os.path.join("random_forest_model.joblib")
VECTORIZER_PATH = os.path.join( "vectorizer.joblib")
LABEL_ENCODER_PATH = os.path.join( "label_encoder.joblib")

# Load the model, vectorizer, and label encoder from the volume
try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    print("Model and supporting files loaded successfully!")
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    raise

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the message from the request
    data = request.json
    message = data.get('message', '')

    if not message:
        return jsonify({'error': 'No message provided'}), 400

    # Transform the message into TF-IDF features
    message_tfidf = vectorizer.transform([message])

    # Make the prediction
    prediction = model.predict(message_tfidf)

    # Decode the prediction (ham or smishing)
    decoded_prediction = label_encoder.inverse_transform(prediction)[0]

    # Return only the label
    return jsonify(decoded_prediction)

# Route to check if the server is running
@app.route('/')
def home():
    return "Flask server is running. Send a POST request to /predict to predict if a message is ham or smishing."

# Start the Flask server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5020)
