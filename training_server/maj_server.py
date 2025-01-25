from flask import Flask, request, jsonify
import pandas as pd
import os
import subprocess

app = Flask(__name__)
VOLUME_PATH = "/shared_volume"
CSV_FILE = os.path.join(VOLUME_PATH, 'messages.csv')
TRAINING_THRESHOLD = 30  # Testing threshold

@app.route('/add_messages', methods=['POST'])
def add_messages():
    data = request.json
    new_messages = data.get('messages', [])
    new_labels = data.get('labels', [])

    if len(new_messages) != len(new_labels):
        return jsonify({"error": "Mismatched messages/labels"}), 400

    new_data = pd.DataFrame({'MainText': new_messages, 'label': new_labels})

    if os.path.exists(CSV_FILE):
        existing_data = pd.read_csv(CSV_FILE)
        updated_data = pd.concat([existing_data, new_data], ignore_index=True)
    else:
        updated_data = new_data

    if updated_data.shape[0] >= TRAINING_THRESHOLD:
        updated_data.to_csv(CSV_FILE, index=False)
        try:
            # Call maj_modele.py instead of train_model.py
            subprocess.run(['python3', 'maj_modele.py'], check=True)
            pd.DataFrame(columns=['MainText', 'label']).to_csv(CSV_FILE, index=False)
            return jsonify({
                "message": f"Model retrained with {len(updated_data)} messages.",
                "total_messages": len(updated_data)
            })
        except subprocess.CalledProcessError as e:
            return jsonify({"error": f"Training failed: {str(e)}"}), 500
    else:
        updated_data.to_csv(CSV_FILE, index=False)
        return jsonify({
            "message": f"{len(new_messages)} messages added. {TRAINING_THRESHOLD - len(updated_data)} remaining.",
            "total_messages": len(updated_data)
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
