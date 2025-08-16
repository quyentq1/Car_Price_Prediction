"""
Car Price Prediction API

This module provides a RESTful API for car price prediction using a pre-trained model.
It's designed to be called from a React frontend.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
import math
from datetime import datetime
from waitress import serve  # Import waitress

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the pre-trained model
MODEL_PATH = 'best_model_RandomForestRegressor.pkl'
model = None

# Define valid values for categorical features
VALID_TRANSMISSIONS = ['Automatic', 'Manual', 'Semi-Auto', 'Other']
VALID_FUEL_TYPES = ['Diesel', 'Hybrid', 'Petrol', 'Other']
VALID_MAKERS = [
    'audi', 'bmw', 'focus', 'ford', 'hyundi', 
    'merc', 'skoda', 'toyota', 'vauxhall', 'vw'
]

def load_model():
    """Load the pre-trained model."""
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
    return model

def preprocess_input(data):
    """Preprocess input data to match the format expected by the model."""
    try:
        print("\nRaw input data:", data)
        
        processed_data = {
            'year': int(data.get('year', 2018)),
            'transmission': str(data.get('transmission', 'Manual')),
            'mileage': float(data.get('mileage', 30000)),
            'fuelType': str(data.get('fuelType', 'Petrol')),
            'tax': float(data.get('tax', 145)),
            'mpg': float(data.get('mpg', 50.0)),
            'engineSize': float(data.get('engineSize', 1.6)),
            'model': str(data.get('model', 'Unknown')),
            'automaker': str(data.get('automaker', 'Unknown')).lower()
        }
        
        df = pd.DataFrame([processed_data])
        
        def get_car_class(automaker):
            luxury_brands = ['audi', 'bmw', 'merc', 'lexus']
            return 'luxury' if automaker in luxury_brands else 'standard'
        
        df['car_class'] = df['automaker'].apply(get_car_class)
        df['ln_mileage'] = np.log1p(df['mileage'].fillna(0).astype(float))
        df['ln_mpg'] = np.log1p(df['mpg'].fillna(0).astype(float))
        
        features = [
            'ln_mileage', 'tax', 'ln_mpg', 'transmission',
            'fuelType', 'automaker', 'car_class'
        ]
        
        print("\nProcessed features:", df[features].to_dict('records'))
        return df[features]
        
    except Exception as e:
        import traceback
        print("\nError in preprocess_input:")
        print(traceback.format_exc())
        raise ValueError(f"Error processing input data: {str(e)}")

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    try:
        data = request.get_json()
        print("\nReceived data:", data)
        
        if not data:
            return jsonify({"error": "No input data provided"}), 400
            
        required_fields = ['year', 'transmission', 'mileage', 'fuelType', 'tax', 'mpg', 'engineSize', 'model', 'automaker']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({"error": f"Missing required fields: {', '.join(missing_fields)}"}), 400
            
        processed_data = preprocess_input(data)
        print("\nProcessed data:", processed_data)
        
        model = load_model()
        prediction = model.predict(processed_data)
        
        gbp_price = round(float(prediction[0]), 2)
        vnd_price = int(gbp_price * 30000)
        formatted_vnd = "{:,}".format(vnd_price).replace(",", ".")
        
        return jsonify({
            "predicted_price_gbp": gbp_price,
            "predicted_price_vnd": formatted_vnd,
            "currency": "VND",
            "model_used": model.named_steps['regressor'].__class__.__name__,
            "exchange_rate": "1 GBP = 30,000 VND",
            "features": {
                'year': data.get('year'),
                'transmission': data.get('transmission'),
                'mileage': data.get('mileage'),
                'fuelType': data.get('fuelType'),
                'tax': data.get('tax'),
                'mpg': data.get('mpg'),
                'engineSize': data.get('engineSize'),
                'model': data.get('model'),
                'automaker': data.get('automaker')
            }
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Prediction failed: {str(e)}',
            'status': 'error'
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    try:
        load_model()
        return jsonify({
            'status': 'healthy',
            'message': 'API is running and model is loaded',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    
    # Run the Flask app using Waitress
    serve(app, host='localhost', port=5001)
