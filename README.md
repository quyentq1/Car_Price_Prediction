# Car Price Prediction API Documentation

This document provides detailed information about the Car Price Prediction API, which allows you to predict car prices based on various features.

## Base URL

```
http://localhost:5000/api
```

## Endpoints

### 1. Health Check

Check if the API is running and the model is loaded.

- **URL**: `/health`
- **Method**: `GET`
- **Response**:
  ```json
  {
    "status": "healthy",
    "message": "API is running and model is loaded",
    "timestamp": "2025-02-16T10:30:45.123456"
  }
  ```

### 2. Predict Car Price

Predict the price of a car based on its features.

- **URL**: `/predict`
- **Method**: `POST`
- **Content-Type**: `application/json`
- **Request Body**:
  ```json
  {
    "year": 2018,
    "transmission": "Automatic",
    "mileage": 25000,
    "fuelType": "Petrol",
    "tax": 150,
    "mpg": 45.6,
    "engineSize": 1.5,
    "automaker": "toyota"
  }
  ```

  | Parameter    | Type   | Required | Description                                    | Valid Values                                      |
  |--------------|--------|----------|------------------------------------------------|---------------------------------------------------|
  | year         | int    | Yes      | Manufacturing year of the car                  | 1990 to current year + 1                          |
  | transmission | string | Yes      | Type of transmission                           | "Automatic", "Manual", "Semi-Auto", "Other"       |
  | mileage      | float  | Yes      | Total miles driven                             | Positive number                                   |
  | fuelType     | string | Yes      | Type of fuel the car uses                      | "Diesel", "Hybrid", "Petrol", "Other"            |
  | tax          | float  | Yes      | Road tax for the car                           | Positive number                                   |
  | mpg          | float  | Yes      | Miles per gallon (fuel efficiency)             | Positive number                                   |
  | engineSize   | float  | Yes      | Size of the engine in liters                   | Positive number                                   |
  | automaker    | string | Yes      | Manufacturer of the car (lowercase)            | "audi", "bmw", "focus", "ford", "hyundi", "merc", "skoda", "toyota", "vauxhall", "vw" |

- **Success Response (200 OK)**:
  ```json
  {
    "predicted_price": 18500.50,
    "status": "success",
    "features": {
      "year": 2018,
      "transmission": "Automatic",
      "mileage": 25000,
      "fuelType": "Petrol",
      "tax": 150,
      "mpg": 45.6,
      "engineSize": 1.5,
      "automaker": "toyota"
    }
  }
  ```

- **Error Response (400 Bad Request)**:
  ```json
  {
    "error": "Invalid input data: Mileage must be a positive number",
    "status": "error"
  }
  ```

- **Error Response (500 Internal Server Error)**:
  ```json
  {
    "error": "Prediction failed: Model file not found",
    "status": "error"
  }
  ```

## Example Usage

### Using cURL

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "year": 2018,
    "transmission": "Automatic",
    "mileage": 25000,
    "fuelType": "Petrol",
    "tax": 150,
    "mpg": 45.6,
    "engineSize": 1.5,
    "automaker": "toyota"
  }'
```

### Using JavaScript (Fetch API)

```javascript
const predictCarPrice = async (carData) => {
  try {
    const response = await fetch('http://localhost:5000/api/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(carData)
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Failed to predict price');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Prediction error:', error);
    throw error;
  }
};

// Example usage
const carData = {
  year: 2018,
  transmission: "Automatic",
  mileage: 25000,
  fuelType: "Petrol",
  tax: 150,
  mpg: 45.6,
  engineSize: 1.5,
  automaker: "toyota"
};

predictCarPrice(carData)
  .then(result => console.log('Predicted price:', result.predicted_price))
  .catch(error => console.error('Error:', error));
```

## Setup Instructions

1. **Install Dependencies**:
   ```bash
   pip install flask flask-cors scikit-learn pandas numpy joblib
   ```

2. **Train the Model**:
   ```bash
   python model_experiments.py
   ```
   This will train the model and save it to the `models` directory.

3. **Run the API**:
   ```bash
   python api.py
   ```
   The API will be available at `http://localhost:5000`

4. **Test the API**:
   ```bash
   # Health check
   curl http://localhost:5000/api/health
   
   # Make a prediction
   curl -X POST http://localhost:5000/api/predict \
     -H "Content-Type: application/json" \
     -d '{"year":2018,"transmission":"Automatic","mileage":25000,"fuelType":"Petrol","tax":150,"mpg":45.6,"engineSize":1.5,"automaker":"toyota"}'
   ```

## Error Handling

The API returns appropriate HTTP status codes along with error messages in the response body when something goes wrong. Always check the status code before processing the response.

## Rate Limiting

Currently, there is no rate limiting implemented. For production use, consider adding rate limiting to prevent abuse.

## CORS

Cross-Origin Resource Sharing (CORS) is enabled for all origins. In a production environment, you might want to restrict this to only allow requests from trusted domains.
