"""
Car Price Prediction - Model Experiments

This script explores different machine learning models for car price prediction 
and implements cross-validation for robust evaluation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
import math
import time

def load_data():
    """
    Load and preprocess the car price data from multiple CSV files.
    
    Returns:
        pd.DataFrame: Processed DataFrame with features and target
    """
    data_dir = 'data/'
    dfs = []
    
    print("Loading data files...")
    # Define required columns and their default values
    required_cols = {
        'year': 0,
        'transmission': 'Unknown',
        'mileage': 0,
        'fuelType': 'Petrol',
        'tax': 0,
        'mpg': 0,
        'engineSize': 0,
        'price': 0,
        'model': 'Unknown'  # Thêm cột model từ dữ liệu gốc
    }
    
    # Load all CSV files from the data directory
    for file in os.listdir(data_dir):
        if not file.endswith('.csv'):
            continue
            
        try:
            file_path = os.path.join(data_dir, file)
            print(f"Loading {file}...")
            
            # Read CSV with error handling
            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                print(f"  - Error reading {file}: {str(e)}")
                continue
            
            # Xử lý cột tax(£) nếu có
            if 'tax(£)' in df.columns:
                df.rename(columns={'tax(£)': 'tax'}, inplace=True)
            
            # Thêm cột automaker từ tên file
            automaker = file.split('.')[0]
            df['automaker'] = automaker
            
            # Thêm các cột bị thiếu với giá trị mặc định
            for col, default_val in required_cols.items():
                if col not in df.columns:
                    df[col] = default_val
                    print(f"  - Added missing column '{col}' with default value: {default_val}")
            
            # Chỉ giữ lại các cột cần thiết
            df = df[list(required_cols.keys()) + ['automaker']]
            
            # Thêm vào danh sách
            dfs.append(df)
            print(f"  - Loaded {len(df)} records")
            
        except Exception as e:
            print(f"  - Error processing {file}: {str(e)}")
            continue
    
    if not dfs:
        raise ValueError("No valid data files found in the data directory")
    
    # Kết hợp tất cả dữ liệu
    df = pd.concat(dfs, ignore_index=True)
    
    # In thông tin cơ bản về dữ liệu
    print(f"\nTotal data shape: {df.shape}")
    print("\nColumns in dataframe:", df.columns.tolist())
    print("\nMissing values per column:")
    print(df.isnull().sum())
    
    # Tạo bản sao để tránh cảnh báo SettingWithCopyWarning
    df = df.copy()
    
    # 1. Xử lý dữ liệu thiếu
    print("\nHandling missing values...")
    for col in ['tax', 'mpg', 'engineSize']:
        if df[col].isna().any():
            median_val = df[col].median()
            print(f"  - Filling {df[col].isna().sum()} missing values in {col} with median: {median_val:.2f}")
            df[col] = df[col].fillna(median_val)
    
    # 2. Áp dụng log transformation cho mileage và mpg
    print("\nApplying log transformations...")
    
    def safe_log(x):
        try:
            return math.log(float(x)) if float(x) > 0 else 0
        except (ValueError, TypeError) as e:
            print(f"  - Warning: Could not convert {x} to log: {str(e)}")
            return 0
    
    df['ln_mileage'] = df['mileage'].astype(float).apply(safe_log)
    df['ln_mpg'] = df['mpg'].astype(float).apply(safe_log)
    
    # 3. Phân nhóm năm sản xuất
    print("\nCategorizing car age...")
    try:
        df['year'] = pd.to_numeric(df['year'])
        df['year_category'] = pd.cut(
            df['year'],
            bins=[0, 2015.9, 2016.9, 2018.9, 2061],
            labels=['over_5_years', '4_to_5_years', '2_to_4_years', 'under_2_years']
        )
    except Exception as e:
        print(f"  - Error categorizing year: {str(e)}")
    
    # 4. Phân nhóm kích thước động cơ
    print("Categorizing engine size...")
    try:
        df['engineSize'] = pd.to_numeric(df['engineSize'])
        engine_bins = [
            -1, 
            df['engineSize'].quantile(0.25)-0.01, 
            df['engineSize'].quantile(0.5)-0.01, 
            df['engineSize'].quantile(0.75)-0.01, 
            df['engineSize'].max()+0.01
        ]
        df['engine_category'] = pd.cut(
            df['engineSize'],
            bins=engine_bins,
            labels=['Small', 'Medium', 'Large', 'Very Large']
        )
    except Exception as e:
        print(f"  - Error categorizing engine size: {str(e)}")
    
    # 5. Xử lý ngoại lệ cho các cột số
    print("\nProcessing numeric columns for outliers...")
    numeric_cols = ['ln_mileage', 'tax', 'ln_mpg', 'price']
    
    for col in numeric_cols:
        if col not in df.columns:
            print(f"  - Warning: Column {col} not found in dataframe")
            continue
            
        try:
            # Chuyển đổi sang số nếu cần
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Tính toán ngưỡng
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR > 0:  # Chỉ xử lý nếu IQR hợp lệ
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # In thông tin để debug
                print(f"  - {col}: Q1={Q1:.2f}, Q3={Q3:.2f}, IQR={IQR:.2f}, "
                      f"Bounds=[{lower_bound:.2f}, {upper_bound:.2f}]")
                
                # Áp dụng cắt giới hạn
                df[col] = np.clip(df[col], lower_bound, upper_bound)
            else:
                print(f"  - {col}: IQR is {IQR:.2f}, skipping outlier detection")
                
        except Exception as e:
            print(f"  - Error processing {col}: {str(e)}")
    
    # Add car class based on automaker
    def get_car_class(automaker):
        luxury = ['merc', 'audi', 'bmw']
        mid_range = ['vw', 'skoda', 'focus']
        
        if automaker in luxury:
            return 'Luxury'
        elif automaker in mid_range:
            return 'Mid-range'
        else:
            return 'Affordable'
    
    df['Class'] = df['automaker'].apply(get_car_class)
    
    # Select features and target
    features = ['year', 'transmission', 'fuelType', 'engineSize', 'Class', 'ln_mileage', 'tax', 'ln_mpg']
    target = 'price'
    
    return df[features + [target]]

def prepare_data(df):
    """
    Prepare data for modeling by splitting into features and target.
    
    Args:
        df (pd.DataFrame): Input DataFrame with features and target
        
    Returns:
        tuple: X (features) and y (target) as numpy arrays
    """
    features = ['year', 'transmission', 'fuelType', 'engineSize', 'Class', 'ln_mileage', 'tax', 'ln_mpg']
    target = 'price'
    
    X = df[features]
    y = df[target]
    
    return X, y

def create_preprocessor():
    """
    Create a preprocessor for numerical and categorical features.
    
    Returns:
        ColumnTransformer: Preprocessor for the model pipeline
    """
    numeric_features = ['ln_mileage', 'tax', 'ln_mpg']
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_features = ['year', 'transmission', 'fuelType', 'engineSize', 'Class']
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on test data and print metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: True target values
    """
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"R² Score: {r2:.4f}")
    
    # Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs Predicted Car Prices')
    plt.show()
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

def train_models():
    """
    Train and evaluate multiple regression models with optimizations for speed.
    
    Returns:
        tuple: (results_df, best_model)
    """
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_data()
    
    # Define required features with default values
    required_columns = {
        'year': 2015,  # Giá trị mặc định là năm trung bình
        'transmission': 'Manual',
        'mileage': 50000,  # Số km trung bình
        'fuelType': 'Petrol',
        'tax': 145,  # Thuế trung bình
        'mpg': 50,  # Mức tiêu thụ nhiên liệu trung bình
        'engineSize': 1.6,  # Dung tích động cơ trung bình
        'price': 15000,  # Giá trung bình
        'automaker': 'Unknown'
    }
    
    # Add missing columns with default values
    print("\nChecking and adding missing columns...")
    for col, default_val in required_columns.items():
        if col not in df.columns:
            print(f"  - Adding missing column '{col}' with default value: {default_val}")
            df[col] = default_val
    
    print("\nCreating features...")
    # Create log features safely with error handling
    try:
        df['ln_mileage'] = np.log1p(df['mileage'].astype(float))
        df['ln_mpg'] = np.log1p(df['mpg'].astype(float))
    except Exception as e:
        print(f"Error creating log features: {str(e)}")
        # Set default values if log transformation fails
        df['ln_mileage'] = np.log1p(50000)  # Giá trị mặc định tương đương 50,000 km
        df['ln_mpg'] = np.log1p(50)  # Giá trị mặc định tương đương 50 mpg
    
    # Add car class based on automaker
    def get_car_class(automaker):
        luxury_brands = ['audi', 'bmw', 'mercedes', 'lexus']
        return 'luxury' if automaker.lower() in luxury_brands else 'standard'
    
    df['car_class'] = df['automaker'].apply(get_car_class)
    
    # Define features and target
    X = df.drop('price', axis=1)
    y = df['price']
    
    # Print columns for debugging
    print("\nAvailable columns in X:", X.columns.tolist())
    
    # Define features for preprocessing
    numeric_features = ['ln_mileage', 'tax', 'ln_mpg']
    categorical_features = ['transmission', 'fuelType', 'automaker', 'car_class']
    
    # Check if all required columns exist
    missing_numeric = [col for col in numeric_features if col not in X.columns]
    missing_categorical = [col for col in categorical_features if col not in X.columns]
    
    if missing_numeric or missing_categorical:
        raise ValueError(f"Missing columns - Numeric: {missing_numeric}, Categorical: {missing_categorical}")
    
    # Split data - using smaller test size for faster training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    
    # Simplify preprocessing
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Use simpler encoding for categories
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ], 
        n_jobs=-1  # Enable parallel processing
    )
    
    # Define models with optimized hyperparameters for speed
    models = {
        'Linear Regression': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression(n_jobs=-1))
        ]),
        'Random Forest': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(
                n_estimators=50,  # Reduced from 100
                max_depth=10,     # Limit tree depth
                min_samples_split=5,  # Increase to reduce overfitting
                n_jobs=-1,       # Use all CPU cores
                random_state=42,
                verbose=1         # Show progress
            ))
        ]),
        'Gradient Boosting': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', GradientBoostingRegressor(
                n_estimators=50,  # Reduced from 100
                learning_rate=0.1,
                max_depth=3,     # Shallower trees for speed
                min_samples_split=5,
                random_state=42,
                verbose=1        # Show progress
            ))
        ])
    }
    
    # Train and evaluate models
    results = []
    best_score = -np.inf
    best_model = None
    
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training {name}...")
        start_time = time.time()
        
        try:
            # Train model with timing
            train_start = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - train_start
            
            # Make predictions
            pred_start = time.time()
            y_pred = model.predict(X_test)
            pred_time = time.time() - pred_start
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Use smaller CV for speed
            print(f"  - Cross-validating {name} (using 3 folds)...")
            cv_scores = cross_val_score(
                model, 
                X, y, 
                cv=3,  # Reduced from 5
                scoring='neg_root_mean_squared_error',
                n_jobs=-1,
                verbose=0  # Reduce verbosity
            )
            cv_rmse = -cv_scores.mean()
            
            # Save results
            model_results = {
                'Model': name,
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'CV_RMSE': cv_rmse,
                'Training Time (s)': train_time,
                'Prediction Time (s)': pred_time
            }
            results.append(model_results)
            
            # Update best model
            if r2 > best_score:
                best_score = r2
                best_model = model
            
            # Print results
            print(f"\n{name} - Test Set Evaluation:")
            print(f"  - MSE: {mse:.2f}")
            print(f"  - RMSE: {rmse:.2f}")
            print(f"  - MAE: {mae:.2f}")
            print(f"  - R²: {r2:.4f}")
            print(f"  - Training Time: {train_time:.2f}s")
            print(f"  - Prediction Time: {pred_time:.4f}s")
            print(f"\n{name} - Cross-validation (3 folds):")
            print(f"  - CV RMSE: {cv_rmse:.2f} (+/- {cv_scores.std() * 2:.2f})")
            
        except Exception as e:
            print(f"\nError training {name}: {str(e)}")
            continue
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save the best model
    if best_model is not None:
        model_name = type(best_model.named_steps['regressor']).__name__
        model_file = f'best_model_{model_name}.pkl'
        joblib.dump(best_model, model_file)
        print(f"\n{'='*50}")
        print(f"Best model ({model_name}) saved as '{model_file}'")
    
    return results_df, best_model

def main():
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Train and evaluate models
    results, best_model = train_models()
    
    print("\nModel training and evaluation complete!")
    print(f"Best model: {best_model}")

if __name__ == "__main__":
    main()
