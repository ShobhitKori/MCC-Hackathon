from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Optional
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import uvicorn
import pickle
import pymongo
import uuid
import time
import datetime
import os
from apscheduler.schedulers.background import BackgroundScheduler

MONGO_URL = "mongodb://localhost:27017/"

# Load the dataset
try:
    df = pd.read_csv('sensor_dataset.csv')
except FileNotFoundError:
    raise RuntimeError("Dataset file 'sensor_dataset.csv' not found. Make sure it's in the same directory as main.py.")
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

# Split data
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.20, random_state=42)

# Model Training
model = XGBClassifier(subsample=0.3, eval_metric='error', n_estimators=500, max_depth=7, learning_rate=0.01, objective='binary:logistic')
model.fit(X_train, y_train)

# Model Saving
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

# Pydantic model for Alert data
class Alert(BaseModel):
    alert_id: Optional[str] = None
    timestamp: datetime.datetime
    equipment_id: str
    message: str
    severity: str
    status: str
    prediction_confidence: Optional[float] = None

# MongoDB Connection
client = pymongo.MongoClient(MONGO_URL)
db = client["sensor"]
sensor_data_collection = db["sensor-dataset"]
alerts_collection = db["alerts"]

# Function to generate alerts from MongoDB data
def generate_alerts_from_mongodb():
    try:
        cursor = sensor_data_collection.find({}, {"_id": 0})
        data = list(cursor)

        if not data:
            print("No data found in MongoDB.")
            return

        input_df = pd.DataFrame(data)
        input_df = input_df[features]
        input_df[features] = scaler.transform(input_df[features])
        predictions = model.predict(input_df)

        for i, prediction in enumerate(predictions):
            if prediction == 1:  # Adjust threshold as needed
                alert_data = {
                    "alert_id": str(uuid.uuid4()),
                    "timestamp": datetime.datetime.now().isoformat(),
                    "equipment_id": f"EQ{i+1}",  # Dynamic equipment ID based on index
                    "message": "High probability of failure detected",
                    "severity": "HIGH",
                    "status": "ACTIVE",
                    "prediction_confidence": int(prediction)
                }
                result = alerts_collection.insert_one(alert_data)
                if not result.acknowledged:
                    print(f"Failed to insert alert for data point {i + 1}")

    except Exception as e:
        print(f"Error generating alerts from MongoDB: {e}")

# Schedule alert generation
scheduler = BackgroundScheduler()
scheduler.add_job(generate_alerts_from_mongodb, 'interval', minutes=1)  # Adjust interval as needed
scheduler.start()

# Alert management endpoints
@app.post("/api/alert")
async def create_alert(alert: Alert):
    try:
        if not alert.alert_id:
            alert.alert_id = str(uuid.uuid4())

        alert_dict = alert.dict()
        alert_dict["timestamp"] = alert_dict["timestamp"].isoformat()
        result = alerts_collection.insert_one(alert_dict)

        if not result.acknowledged:
            raise HTTPException(status_code=500, detail="Failed to create alert")

        return JSONResponse(content={"message": "Alert created successfully", "alert_id": alert.alert_id}, status_code=201)
    except Exception as e:
        print(f"Error creating alert: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

# @app.put("/api/alert/{alert_id}")
# async def update_alert(alert_id: str, alert_data: Alert):
#     try:
#         alert_dict = alert_data.dict()
#         alert_dict["timestamp"] = alert_dict["timestamp"].isoformat()
#         result = alerts_collection.update_one({"alert_id": alert_id}, {"$set": alert_dict})

#         if result.modified_count == 0:
#             raise HTTPException(status_code=404, detail="Alert not found")

#         return JSONResponse(content={"message": "Alert updated successfully"}, status_code=200)
#     except Exception as e:
#         print(f"Error updating alert: {e}")
#         return JSONResponse(content={"error": str(e)}, status_code=500)
    
@app.get("/api/alert/{alert_id}")
async def get_alert_by_id(alert_id: str):
    try:
        # Find the alert in MongoDB
        alert_data = alerts_collection.find_one({"alert_id": alert_id})

        if alert_data is None:
            raise HTTPException(status_code=404, detail="Alert not found")

        # Extract the alert message
        alert_message = alert_data.get("message")

        if alert_message is None:
            raise HTTPException(status_code=500, detail="Alert message not found")

        return JSONResponse(content={"alert_message": alert_message}, status_code=200)
    except Exception as e:
        print(f"Error retrieving alert: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    print(f"Request: {request.method} {request.url} - {response.status_code} - {process_time:.4f}s")
    return response

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)