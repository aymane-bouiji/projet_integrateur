from flask import Flask, request, jsonify
import pandas as pd
import os
import subprocess





# Initialize Flask app
app = Flask(__name__)

# Define the path to the volume
VOLUME_PATH = "/shared_volume"

# Path to the CSV file for storing messages
CSV_FILE = os.path.join(VOLUME_PATH, 'messages.csv')

# Threshold to trigger model training
TRAINING_THRESHOLD = 30  # Number of messages required for training

# Route to receive new messages and labels
@app.route('/add_messages', methods=['POST'])
def add_messages():
    # Get data from the request
    data = request.json
    new_messages = data.get('messages', [])  # List of messages
    new_labels = data.get('labels', [])      # List of corresponding labels

    # Check if the number of messages and labels match
    if len(new_messages) != len(new_labels):
        return jsonify({"error": "The number of messages and labels does not match."}), 400

    # Create a DataFrame with the new messages and labels
    new_data = pd.DataFrame({
        'MainText': new_messages,
        'label': new_labels
    })

    # If the CSV file exists, load the existing data
    if os.path.exists(CSV_FILE):
        existing_data = pd.read_csv(CSV_FILE)
        updated_data = pd.concat([existing_data, new_data], ignore_index=True)
    else:
        updated_data = new_data

    # Check if the training threshold is reached or exceeded
    if len(updated_data) >= TRAINING_THRESHOLD:
        # Save the updated data to the CSV file (temporarily)
        updated_data.to_csv(CSV_FILE, index=False)
      
        

        # Run the training script
        try:
            subprocess.run(['python3', 'train_model.py'], check=True)
            # Reset the CSV file after training
            pd.DataFrame(columns=['MainText', 'label']).to_csv(CSV_FILE, index=False)
            return jsonify({
                "message": f"{len(new_messages)} messages were added successfully. The model was retrained with {len(updated_data)} messages. The messages.csv file has been cleared.",
                "total_messages": len(updated_data)
            })
        except subprocess.CalledProcessError as e:
            return jsonify({"error": f"Error during model training: {str(e)}"}), 500
    else:
        # Save the updated data to the CSV file
        updated_data.to_csv(CSV_FILE, index=False)
        return jsonify({
            "message": f"{len(new_messages)} messages were added successfully. {TRAINING_THRESHOLD - len(updated_data)} more messages are needed for training.",
            "total_messages": len(updated_data)
        })

        
# Start the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
