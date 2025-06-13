# app.py
from flask import Flask, render_template, request
import numpy as np
import joblib
import os

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model
MODEL_PATH = os.path.join('models', 'house_price_model.pkl')
model = joblib.load(MODEL_PATH)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        input_array = np.array([features])
        prediction = model.predict(input_array)
        result = round(prediction[0], 2)
        return render_template('index.html', prediction_text=f"Estimated House Price: ${result}K")
    except Exception:
        return render_template('index.html', prediction_text="\u274C Invalid input. Please enter valid numerical values.")

if __name__ == '__main__':
    app.run(debug=True)
