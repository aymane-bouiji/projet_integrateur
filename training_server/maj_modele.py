import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from joblib import dump, load
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
import shutil

# Télécharger les ressources nécessaires de NLTK
nltk.download('stopwords')
nltk.download('wordnet')

# Initialisation des outils de nettoyage
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def Clean_message(text):
    """Nettoyage du texte"""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    # Supprimer les stop words
    text = ' '.join(word for word in text.split() if word not in stop_words)
    # Lemmatization
    text = ' '.join(lemmatizer.lemmatize(word) for word in text.split())
    return text

# Charger les fichiers existants (s'ils existent)
try:
    vectorizer = load('vectorizer.joblib')
    model = load('random_forest_model.joblib')
    label_encoder = load('label_encoder.joblib')
except FileNotFoundError:
    # Initialiser de nouveaux objets si les fichiers n'existent pas
    vectorizer = CountVectorizer()
    model = RandomForestClassifier()
    label_encoder = LabelEncoder()

# Charger les messages depuis le fichier CSV
CSV_FILE = 'messages.csv'
if not os.path.exists(CSV_FILE):
    print("Aucun message trouvé. Veuillez d'abord ajouter des messages via le serveur Flask.")
    exit()

dataset = pd.read_csv(CSV_FILE)

# Nettoyer les messages
dataset['MainText'] = dataset['MainText'].apply(Clean_message)

# Remplacer tous les autres labels par 'phishing' (au cas où il y en aurait)
dataset['label'] = dataset['label'].replace({'smishing': 'phishing', 'spam': 'phishing', 'malware': 'phishing'})

# Filtrer pour ne garder que les labels 'ham' et 'phishing'
dataset = dataset[dataset['label'].isin(['ham', 'phishing'])]

# Mélanger les données du dataset
dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

# Vectoriser les SMS
X = vectorizer.fit_transform(dataset['MainText'])

# Encoder les labels
y = label_encoder.fit_transform(dataset['label'])

# Ré-entraîner le modèle
model.fit(X, y)

# Sauvegarder les fichiers localement d'abord
dump(vectorizer, 'vectorizer.joblib')
dump(model, 'random_forest_model.joblib')
dump(label_encoder, 'label_encoder.joblib')

# Tentative de copie vers le volume partagé
SHARED_VOLUME_PATH = '/shared_volume'
try:
    model_files = ['vectorizer.joblib', 'random_forest_model.joblib', 'label_encoder.joblib']
    for file in model_files:
        shared_file_path = os.path.join(SHARED_VOLUME_PATH, file)
        if os.path.exists(SHARED_VOLUME_PATH):  # Only try to copy if the shared volume exists
            shutil.copy2(file, shared_file_path)
            print(f"Fichier {file} copié vers le volume partagé avec succès.")
        else:
            print(f"Le volume partagé n'est pas accessible. Les fichiers sont sauvegardés localement uniquement.")
            break
    
    print("Le modèle a été mis à jour avec succès.")
except (PermissionError, OSError) as e:
    print(f"Impossible de copier vers le volume partagé: {str(e)}")
    print("Les fichiers ont été sauvegardés localement uniquement.")
