import os
import torch
import firebase_admin
from flask import Flask, request, jsonify
from firebase_admin import credentials, firestore
import torchvision.models as models

# Constants
FIREBASE_CREDENTIALS_PATH = "/home/mich02/Desktop/UMKC_DS_Capstone/firebase_service_account_key.json"
MODEL_WEIGHTS_PATH = "/home/mich02/Desktop/UMKC_DS_Capstone/facial_stroke_model.pth"

# Initialize Flask app
app = Flask(__name__)

# Initialize Firebase Admin SDK
def initialize_firebase():
    cred = credentials.Certificate(FIREBASE_CREDENTIALS_PATH)
    firebase_admin.initialize_app(cred)

# Load and prepare the model
def load_model():
    model = models.resnet50()
    model.fc = torch.nn.Linear(in_features=2048, out_features=2)
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH))
    model.eval()
    return model

# Initialize Firebase and model
initialize_firebase()
db = firestore.client()
model = load_model()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if not data or 'input' not in data:
        return jsonify({'error': 'Input data not provided'}), 400

    try:
        # Convert input to torch tensor
        inputs = torch.tensor([data['input']], dtype=torch.float32)

        # Make a prediction
        with torch.no_grad():
            prediction = model(inputs).item()

        # Save prediction to Firestore
        db.collection('predictions').add({'prediction': prediction})

        return jsonify({'prediction': prediction}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
