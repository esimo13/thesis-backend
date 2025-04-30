import os
import numpy as np
import joblib
import tensorflow as tf
from flask import Flask, request, jsonify
from preprocess import preprocess_input_txt
from flask_cors import CORS  # Import CORS
import logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}},)

# Check if model files exist, if not, run model.py
if not os.path.exists("scaler.pkl") or not os.path.exists("svm_model.pkl") or not os.path.exists("gru_model.keras"):
    print("Model files missing. Training models now...")
    os.system("python model.py")  # Automatically runs model.py to train models

# Load trained models after ensuring they exist
svm_model = joblib.load("svm_model.pkl")
gru_model = tf.keras.models.load_model("gru_model.keras")
scaler = joblib.load("scaler.pkl")



UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def home():
    return "EEG Seizure Detection API is Running!"
@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"message": "pong"}), 200
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Preprocess EEG data
    input_data = preprocess_input_txt(file_path)

    # Get Predictions
    svm_prob = svm_model.predict_proba(input_data.reshape(1, -1))
    input_data = input_data.reshape((1, 1, 1025))  # Reshape to match GRU expected shape
    gru_prob = gru_model.predict(input_data)

    # Ensemble (Averaging Probabilities)
    final_prob = (svm_prob + gru_prob) / 2
    predicted_class = np.argmax(final_prob, axis=1)[0]

    # Class Mapping
    class_mapping = {0: "Interictal (Normal brain activity)", 
                     1: "Preictal (Seizure warning)", 
                     2: "Ictal (Seizure occurring)"}
    
    return jsonify({"prediction": class_mapping[predicted_class]})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
