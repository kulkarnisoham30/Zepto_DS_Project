from flask import Flask, request, jsonify
import pickle
import numpy as np
import traceback

app = Flask(__name__)

# Load trained model
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)


@app.route('/')
def home():
    return jsonify({"message": "ML Model API is running!"})


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)
        prediction = model.predict(features)
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        # Print full traceback to container logs for debugging
        print('Exception in /predict:', e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
