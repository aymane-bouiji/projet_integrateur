import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from joblib import dump, load
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def Clean_message(text):
    """Nettoyage du texte"""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    
    # Keep numbers, URLs, and critical keywords (e.g., "urgent", "free", "http")
    text = ' '.join(
        word for word in text.split() 
        if (word not in stop_words) or word.isdigit() or ("http" in word)
    )
    
    # Lemmatization
    text = ' '.join(lemmatizer.lemmatize(word) for word in text.split())
    return text

try:
    vectorizer = load('/shared_volume/vectorizer.joblib')
    model = load('/shared_volume/random_forest_model.joblib')
    label_encoder = load('/shared_volume/label_encoder.joblib')
except FileNotFoundError:
    vectorizer = CountVectorizer(token_pattern=r'(?u)\b\w+\b')  # Allow single-character tokens
    model = RandomForestClassifier()
    label_encoder = LabelEncoder()

CSV_FILE = '/shared_volume/messages.csv'
if not os.path.exists(CSV_FILE):
    raise FileNotFoundError("No messages found in shared volume!")

dataset = pd.read_csv(CSV_FILE)
dataset['MainText'] = dataset['MainText'].apply(Clean_message)

# Debug: Print cleaned messages
print("Sample cleaned messages after preprocessing:")
print(dataset['MainText'].head())

dataset['label'] = dataset['label'].replace({'smishing': 'phishing', 'spam': 'phishing', 'malware': 'phishing'})
dataset = dataset[dataset['label'].isin(['ham', 'phishing'])]
dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

X = vectorizer.fit_transform(dataset['MainText'])
y = label_encoder.fit_transform(dataset['label'])
model.fit(X, y)

dump(vectorizer, '/shared_volume/vectorizer.joblib')
dump(model, '/shared_volume/random_forest_model.joblib')
dump(label_encoder, '/shared_volume/label_encoder.joblib')

print("Le modèle a été mis à jour avec succès.")
