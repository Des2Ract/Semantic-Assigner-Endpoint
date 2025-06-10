from flask import Flask, request, jsonify
from predict_tags import load_model_and_encoders, predict_tag, post_process_tags
import json

app = Flask(__name__)

# Load models once on startup
model, label_encoder, ohe, imputer, scaler, multi_classifier = load_model_and_encoders()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        predict_tag(data, 0, None, None, None, None, model, label_encoder, ohe, imputer, scaler, multi_classifier)
        result = post_process_tags(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
