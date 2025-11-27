from flask import Flask, request, jsonify 
import pandas as pd
import joblib

app = Flask(__name__)

# Load the retrained model
model_api = joblib.load('logistic_regression_model.joblib')

# Load the scaler
scaler_api = joblib.load('scaler.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_df = pd.DataFrame([data])

    scaled_input = scaler_api.transform(input_df)

    prediction = model_api.predict(scaled_input)
    prediction_proba = model_api.predict_proba(scaled_input)

    return jsonify({
        'prediction': int(prediction[0]),
        'probability_no_diabetes': float(prediction_proba[0][0]),
        'probability_diabetes': float(prediction_proba[0][1])
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
