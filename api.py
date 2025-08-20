from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
import re
from datetime import datetime

app = Flask(__name__)
CORS(app)

MODEL_PATH = 'best_model_RandomForestRegressor.pkl'
VALID_VALUES_PATH = 'models/valid_values.pkl'
DATA_PATH = 'data.csv'

model, valid_values = None, None

# ======================== LOAD MODEL & VALID VALUES ========================
def load_model():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
    return model

def load_valid_values():
    global valid_values
    if valid_values is None:
        if not os.path.exists(VALID_VALUES_PATH):
            raise FileNotFoundError(f"Valid values file not found at {VALID_VALUES_PATH}")
        valid_values = joblib.load(VALID_VALUES_PATH)
    return valid_values

# ======================== PARSE HELPERS ========================
def parse_price(price_str):
    if pd.isna(price_str):
        return np.nan
    price_str = price_str.replace(' ', '').lower()
    parts = re.findall(r'(\d+)(billionvnd|millionvnd)', price_str)
    total_price = 0
    for value, unit in parts:
        value = int(value)
        if unit == 'billionvnd':
            total_price += value * 1_000_000_000
        elif unit == 'millionvnd':
            total_price += value * 1_000_000
    return total_price

def parse_mileage(mileage_str):
    if pd.isna(mileage_str):
        return np.nan
    mileage_str = str(mileage_str).replace(',', '').replace('km', '').strip()
    return int(mileage_str) if mileage_str.isdigit() else np.nan

def parse_engine(engine_str):
    """
    Parses the engine string for fuel type and engine size.
    This version works with English fuel types.
    """
    if pd.isna(engine_str):
        return ('Unknown', np.nan)
    engine_str = str(engine_str).lower()
    
    # Updated map for English fuel types
    fuel_type_map = {
        'petrol': 'Petrol', 
        'diesel': 'Diesel', 
        'electric': 'Electric', 
        'hybrid': 'Hybrid'
    }
    
    fuel_type = 'Unknown'
    for key, value in fuel_type_map.items():
        if key in engine_str:
            fuel_type = value
            break
            
    engine_size_match = re.search(r'(\d+\.\d+)', engine_str)
    engine_size = float(engine_size_match.group(1)) if engine_size_match else np.nan
    return (fuel_type, engine_size)

def parse_name(name_str):
    if pd.isna(name_str):
        return ('Unknown', 'Unknown')
    parts = str(name_str).split()
    automaker = parts[0]
    model = " ".join(parts[1:])
    return (automaker, model)

# ======================== API ENDPOINTS ========================
@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        load_model()
        json_data = request.get_json()
        
        df = pd.DataFrame(json_data, index=[0])
        
        required_features = ['year', 'mileage', 'fuelType', 'engineSize', 'transmission', 'automaker', 'model', 'condition', 'origin', 'bodyStyle']
        for col in required_features:
            if col not in df.columns:
                return jsonify({'error': f'Missing required feature: {col}', 'status': 'error'}), 400

        prediction = model.predict(df)
        
        return jsonify({'status': 'success', 'prediction': prediction[0]})
        
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/get_unique_values', methods=['GET'])
def get_unique_values():
    try:
        df = pd.read_csv(DATA_PATH)
        df[['automaker', 'model']] = df['Name'].apply(lambda x: pd.Series(parse_name(x)))
        df[['fuelType', 'engineSize']] = df['Engine'].apply(lambda x: pd.Series(parse_engine(x)))
        
        brands = sorted(df['automaker'].dropna().unique().tolist())
        brand_models = {brand: sorted(df[df['automaker'] == brand]['model'].dropna().unique().tolist()) for brand in brands}
        fuel_types = sorted(df['fuelType'].dropna().unique().tolist())
        
        return jsonify({'status': 'success','data': {'brands': brands,'brand_models': brand_models,'fuel_types': fuel_types}})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/get_models_by_brand/<brand>', methods=['GET'])
def get_models_by_brand(brand):
    try:
        df = pd.read_csv(DATA_PATH)
        df[['automaker','model']] = df['Name'].apply(lambda x: pd.Series(parse_name(x)))
        models = sorted(df[df['automaker']==brand]['model'].dropna().unique().tolist())
        return jsonify({'status': 'success','brand': brand,'models': models})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    try:
        load_model(); load_valid_values()
        return jsonify({'status': 'healthy','timestamp': datetime.now().isoformat()})
    except Exception as e:
        return jsonify({'status': 'unhealthy','error': str(e)}), 500

@app.route('/api/metadata', methods=['GET'])
def get_metadata():
    try:
        valid = load_valid_values()
        return jsonify({"required_features": ['year','mileage','fuelType','engineSize','transmission','automaker','model','condition','origin','bodyStyle'],"valid_values": valid})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)