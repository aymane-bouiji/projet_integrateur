from flask import Flask, request, jsonify
import pandas as pd
import os
import subprocess

# Initialisation de l'application Flask
app = Flask(__name__)

# Chemin du fichier CSV pour stocker les messages
CSV_FILE = 'messages.csv'

# Seuil pour déclencher l'entraînement du modèle
TRAINING_THRESHOLD = 2  # Nombre de messages nécessaires pour l'entraînement

# Route pour recevoir de nouveaux messages et labels
@app.route('/add_messages', methods=['POST'])
def add_messages():
    # Récupérer les données de la requête
    data = request.json
    new_messages = data.get('messages', [])  # Liste de messages
    new_labels = data.get('labels', [])      # Liste de labels correspondants

    # Vérifier que les messages et les labels ont la même longueur
    if len(new_messages) != len(new_labels):
        return jsonify({"error": "Le nombre de messages et de labels ne correspond pas."}), 400

    # Créer un DataFrame avec les nouveaux messages et labels
    new_data = pd.DataFrame({
        'MainText': new_messages,
        'label': new_labels
    })

    # Si le fichier CSV existe, charger les données existantes
    if os.path.exists(CSV_FILE):
        existing_data = pd.read_csv(CSV_FILE)
        updated_data = pd.concat([existing_data, new_data], ignore_index=True)
    else:
        updated_data = new_data

    # Sauvegarder les données mises à jour dans le fichier CSV
    updated_data.to_csv(CSV_FILE, index=False)

    # Vérifier si le seuil pour l'entraînement est atteint
    if len(updated_data) >= TRAINING_THRESHOLD:
        # Lancer le script d'entraînement
        try:
            subprocess.run(['python3', 'maj_modele.py'], check=True)
            return jsonify({
                "message": f"{len(new_messages)} messages ont été ajoutés avec succès. Le modèle a été ré-entraîné.",
                "total_messages": len(updated_data)
            })
        except subprocess.CalledProcessError as e:
            return jsonify({"error": f"Erreur lors de l'entraînement du modèle : {str(e)}"}), 500
    else:
        return jsonify({
            "message": f"{len(new_messages)} messages ont été ajoutés avec succès. {TRAINING_THRESHOLD - len(updated_data)} messages supplémentaires nécessaires pour l'entraînement.",
            "total_messages": len(updated_data)
        })

# Démarrer l'application Flask
if __name__ == '__main__':
    app.run(debug=True,port="5011",host='0.0.0.0')
