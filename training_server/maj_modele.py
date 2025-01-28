import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from joblib import dump, load
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
import sys

# Define paths
LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))  # Local directory of the script
PV_DIR = "/shared_volume"                               # Persistent Volume directory

# Create PV_DIR if it doesn't exist
try:
    os.makedirs(PV_DIR, exist_ok=True)
    print(f"Ensuring export directory exists: {PV_DIR}")
except PermissionError:
    print(f"Error: No permission to create directory {PV_DIR}")
    print("Please ensure you have the correct permissions or the directory exists")
    sys.exit(1)

# NLTK setup
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def Clean_message(text):
    """Clean text identically across both services"""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = ' '.join(word for word in text.split() if word not in stop_words)
    text = ' '.join(lemmatizer.lemmatize(word) for word in text.split())
    return text

# Check for local CSV first
LOCAL_CSV = os.path.join(LOCAL_DIR, 'messages.csv')
if not os.path.exists(LOCAL_CSV):
    print(f"Error: Local messages.csv not found at {LOCAL_CSV}")
    print("Please add it to the training directory.")
    sys.exit(1)

# Load existing models from LOCAL_DIR with error handling
try:
    print("Attempting to load existing models from local directory...")
    vectorizer = load(os.path.join(LOCAL_DIR, 'vectorizer.joblib'))
    model = load(os.path.join(LOCAL_DIR, 'random_forest_model.joblib'))
    label_encoder = load(os.path.join(LOCAL_DIR, 'label_encoder.joblib'))
    print("Successfully loaded existing models")
except FileNotFoundError:
    print("No existing models found. Initializing new models...")
    vectorizer = CountVectorizer()
    model = RandomForestClassifier()
    label_encoder = LabelEncoder()

# Process data
print("Loading and processing dataset...")
dataset = pd.read_csv(LOCAL_CSV)
dataset['MainText'] = dataset['MainText'].apply(Clean_message)
dataset['label'] = dataset['label'].replace(
    {'smishing': 'phishing', 'spam': 'phishing', 'malware': 'phishing'}
)
dataset = dataset[dataset['label'].isin(['ham', 'phishing'])].sample(frac=1, random_state=42)

# Train model
print("Training model...")
X = vectorizer.fit_transform(dataset['MainText'])
y = label_encoder.fit_transform(dataset['label'])
model.fit(X, y)

# Save updated models and processed dataset to PV with error handling
try:
    print(f"Saving models and processed dataset to {PV_DIR}...")
    dump(vectorizer, os.path.join(PV_DIR, 'vectorizer.joblib'))
    dump(model, os.path.join(PV_DIR, 'random_forest_model.joblib'))
    dump(label_encoder, os.path.join(PV_DIR, 'label_encoder.joblib'))
    dataset.to_csv(os.path.join(PV_DIR, 'messages_processed.csv'), index=False)
    print("Successfully saved all files to Persistent Volume!")
except PermissionError:
    print(f"Error: No permission to write to {PV_DIR}")
    print("Please check directory permissions")
    sys.exit(1)
except Exception as e:
    print(f"Error saving files: {str(e)}")
    sys.exit(1)
