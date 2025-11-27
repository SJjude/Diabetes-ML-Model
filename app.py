from flask import Flask, request, jsonify
import pandas as pd

app = Flask(__name__)

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
    app.run(debug=True)
