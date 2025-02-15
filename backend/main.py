from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from typing import List
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd  
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier
import uvicorn
import pickle
import pymongo

import time

try:
    df = pd.read_csv('sensor_dataset.csv')
except FileNotFoundError:
    raise RuntimeError("Dataset file 'your_dataset.csv' not found.")
except Exception as e:
    raise RuntimeError(f"Error loading dataset: {e}")

# Data preprocessing
df.dropna(inplace=True)


# Feature Selection
features = ['footfall', 'tempMode', 'AQ', 'USS', 'CS', 'VOC', 'RP', 'IP', 'Temperature']
target = 'fail'

# Feature Scaling
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.20, random_state=42) 

# Model Training 
model = XGBClassifier(subsample = 0.3, eval_metric = 'error', n_estimators=500, max_depth=7, learning_rate=0.01, objective='binary:logistic') 
model.fit(X_train, y_train)

# Model saving
try:
    pickle.dump(model, open("xgboost.pkl", "wb"))  
    print("Model saved successfully.")
except Exception as e:
    raise RuntimeError(f"Error saving model: {e}")

# Load the trained model
try:
    model = pickle.load(open('xgboost.pkl', "rb"))
    print("Model loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

app = FastAPI()

class PredictionDataPoint(BaseModel):
    footfall: int
    tempMode: int
    AQ: int
    USS: int
    CS: int
    VOC: int
    RP: int
    IP: int
    Temperature: int

# MongoDB Configuration
MONGO_URL = "mongodb://localhost:27017/"  
client = pymongo.MongoClient(MONGO_URL)
db = client["sensor-database"] 
collection = db["sensor_data"]

@app.post("/api/predict")
async def predict(data: List[PredictionDataPoint]):
    try:
        predictions = []
        for data_point in data:
            data_dict = data_point.dict()
            input_df = pd.DataFrame([data_dict])
            print(input_df)
            missing_features = set(features) - set(input_df.columns)
            if missing_features:
                raise ValueError(f"Missing features: {list(missing_features)}")
            input_df = input_df[features]

            for col in features:
                if input_df[col].dtype == 'object':
                    try:
                        input_df[col] = pd.to_numeric(input_df[col])
                    except ValueError:
                        raise ValueError(f"Cannot convert column '{col}' to numeric.")
            
            prediction = model.predict(input_df)
            predictions.append(int(prediction))

        return JSONResponse(content={"predictions": predictions}, status_code=200)
    except Exception as e:
        print(f"Prediction Error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Add Middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    print(f"Request: {request.method} {request.url} - {response.status_code} - {process_time:.4f}s")
    return response

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)