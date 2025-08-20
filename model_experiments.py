import pandas as pd
import numpy as np
import re
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import time

def parse_price(price_str):
    """
    Parses a price string in English and converts it to a numerical value.
    Example: "3 Billion VND 250 Million VND" -> 3,250,000,000
    """
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
    """
    Parses a mileage string like '3,800 Km' -> 3800
    """
    if pd.isna(mileage_str):
        return np.nan
    
    mileage_str = mileage_str.replace(',', '').replace('Km', '').strip()
    
    try:
        return float(mileage_str)
    except ValueError:
        return np.nan

def parse_engine(engine_str):
    """
    Parses engine string to extract fuel type and size.
    Example: "Petrol 2.0 L" -> ('Petrol', 2.0)
    """
    if pd.isna(engine_str):
        return ('Unknown', np.nan)
        
    engine_str = engine_str.lower()
    fuel_type_map = {
        'petrol': 'Petrol', 
        'diesel': 'Diesel',
        'electric': 'Electric',
        'hybrid': 'Hybrid'
    }
    
    fuel_type = 'Unknown'
    for key, val in fuel_type_map.items():
        if key in engine_str:
            fuel_type = val
            break
            
    engine_size_match = re.search(r'(\d+\.?\d*)\s*l', engine_str)
    engine_size = float(engine_size_match.group(1)) if engine_size_match else np.nan
    
    return fuel_type, engine_size

def parse_name(name_str):
    """
    Extract automaker and model from name
    Example: "Toyota Veloz Cross 1.5 CVT - 2024" -> ('Toyota', 'Veloz Cross')
    """
    if pd.isna(name_str):
        return ('Unknown', 'Unknown')
        
    parts = name_str.split(' ')
    automaker = parts[0]
    model = ' '.join(parts[1:3])
    
    return automaker, model

def parse_description(desc_str):
    """
    Extract transmission from description.
    """
    if pd.isna(desc_str):
        return 'Unknown'
        
    desc_str = desc_str.lower()
    if 'automatic transmission' in desc_str:
        return 'Automatic'
    elif 'manual transmission' in desc_str:
        return 'Manual'
    else:
        return 'Unknown'
        
def load_data(file_path='data.csv'):
    """
    Load and preprocess data.
    """
    print(f"Loading and preprocessing data from {file_path}...")
    df = pd.read_csv(file_path)

    # Rename to match English headers
    df = df.rename(columns={
        'Year of manufacture': 'year',
        'Condition': 'condition',
        'Origin': 'origin',
        'Body type': 'bodyStyle',
        'Engine': 'engine',
        'Mileage': 'mileage_text',
        'Price': 'price_text',
        'Name': 'name',
        'Description': 'description'
    })
    
    # Apply parsing
    df['price'] = df['price_text'].apply(parse_price)
    df['mileage'] = df['mileage_text'].apply(parse_mileage)
    
    engine_parsed = df['engine'].apply(parse_engine)
    df['fuelType'] = [x[0] for x in engine_parsed]
    df['engineSize'] = [x[1] for x in engine_parsed]
    
    name_parsed = df['name'].apply(parse_name)
    df['automaker'] = [x[0] for x in name_parsed]
    df['model'] = [x[1] for x in name_parsed]
    
    df['transmission'] = df['description'].apply(parse_description)

    # Drop rows with missing price
    df.dropna(subset=['price'], inplace=True)
    
    # Drop unused columns
    df = df.drop(columns=[
        'Link', 'Car_code', 'Location', 'price_text', 'engine', 
        'name', 'mileage_text', 'description', 
        'Exterior color', 'Interior color', 'Seats', 'Doors'
    ], errors='ignore')
    
    required_cols = [
        'year', 'mileage', 'fuelType', 'engineSize', 
        'transmission', 'automaker', 'model',
        'condition', 'origin', 'bodyStyle', 'price'
    ]
    df = df[required_cols]

    return df

def create_model_pipeline():
    numeric_features = ['year', 'mileage', 'engineSize']
    categorical_features = ['fuelType', 'transmission', 'automaker', 'model', 
                            'condition', 'origin', 'bodyStyle']
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
        
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    
    return model

def run_experiments():
    df = load_data()
    
    X = df.drop('price', axis=1)
    y = df['price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = create_model_pipeline()
    
    print("Training model...")
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    print("Evaluating model on test set...")
    start_time = time.time()
    y_pred = model.predict(X_test)
    pred_time = time.time() - start_time
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\nModel Evaluation Results:")
    print(f"  - MSE: {mse:.2f}")
    print(f"  - RMSE: {rmse:.2f}")
    print(f"  - MAE: {mae:.2f}")
    print(f"  - R²: {r2:.4f}")
    print(f"  - Training Time: {train_time:.2f}s")
    print(f"  - Prediction Time: {pred_time:.4f}s")
    
    model_name = type(model.named_steps['regressor']).__name__
    model_file = f'best_model_{model_name}.pkl'
    joblib.dump(model, model_file)
    print(f"\nModel saved as '{model_file}'")
    
    return model, df

if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    _, df_full = run_experiments()

    # Thêm .str.strip() để đảm bảo không có khoảng trắng thừa
    df_full['automaker'] = df_full['automaker'].str.strip()
    df_full['model'] = df_full['model'].str.strip()

    print("Creating valid values for API...")
    valid_data = {
        'VALID_FUEL_TYPES': sorted(list(df_full['fuelType'].dropna().unique())),
        'VALID_TRANSMISSIONS': sorted(list(df_full['transmission'].dropna().unique())),
        'VALID_MAKERS': sorted(list(df_full['automaker'].dropna().unique())),
        
        # SỬA LỖI CHÍNH: Dùng groupby() để nhóm model theo hãng xe
        'VALID_MODELS': df_full.groupby('automaker')['model'].unique().apply(list).to_dict(),
        
        'VALID_CONDITIONS': sorted(list(df_full['condition'].dropna().unique())),
        'VALID_ORIGINS': sorted(list(df_full['origin'].dropna().unique())),
        
        # Sửa lỗi tên key 'VALID_BODY_STYLES' -> 'VALID_BODYSTYLES'
        'VALID_BODYSTYLES': sorted(list(df_full['bodyStyle'].dropna().unique())),
    }
    
    # Sửa lỗi cách dùng joblib.dump
    valid_values_file = 'models/valid_values.pkl'
    joblib.dump(valid_data, valid_values_file)
    
    print(f"\nValid values saved to '{valid_values_file}'")