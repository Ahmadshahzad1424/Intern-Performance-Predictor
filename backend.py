import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import os
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("InternPredictor")

app = FastAPI(title="Internee.pk Performance Predictor API")

# Mount static files for the UI
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Mock Model Storage (In real world, we'd load a saved .json model)
class ModelWrapper:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False

    def train_initial_model(self):
        logger.info("Training baseline model...")
        np.random.seed(42)
        n_samples = 1000
        time = np.random.uniform(10, 100, n_samples)
        feedback = np.random.uniform(1, 5, n_samples)
        attendance = np.random.uniform(0.7, 1.0, n_samples)
        
        # Target: Performance Score
        score = (100 - time) * 0.4 + (feedback * 15) + (attendance * 20)
        score = np.clip(score + np.random.normal(0, 5, n_samples), 0, 100)
        
        X = pd.DataFrame({'time': time, 'feedback': feedback, 'attendance': attendance})
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        self.model = xgb.XGBRegressor(n_estimators=100)
        self.model.fit(X_scaled, score)
        self.is_trained = True
        logger.info("Baseline model ready.")

    def predict(self, time: float, feedback: float, attendance: float):
        if not self.is_trained:
            self.train_initial_model()
        X = pd.DataFrame([[time, feedback, attendance]], columns=['time', 'feedback', 'attendance'])
        X_scaled = self.scaler.transform(X)
        prediction = self.model.predict(X_scaled)[0]
        return float(np.clip(prediction, 0, 100))

model_engine = ModelWrapper()

# Request Model
class InternData(BaseModel):
    task_completion_time: float
    feedback_rating: float
    attendance_rate: float

@app.get("/")
async def read_index():
    from fastapi.responses import FileResponse
    return FileResponse("static/index.html")

@app.post("/predict")
async def predict_performance(data: InternData):
    try:
        prediction = model_engine.predict(
            data.task_completion_time,
            data.feedback_rating,
            data.attendance_rate
        )
        
        # Risk Analysis
        status = "Excellent" if prediction > 80 else "Stable" if prediction > 60 else "At Risk"
        recommendation = "Promote to Junior Role" if status == "Excellent" else "Continue Training" if status == "Stable" else "Requires Urgent Mentorship"
        
        return {
            "score": round(prediction, 2),
            "status": status,
            "recommendation": recommendation,
            "metrics": {
                "efficiency": round((100 - data.task_completion_time), 2),
                "satisfaction": data.feedback_rating
            }
        }
    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    model_engine.train_initial_model()
    uvicorn.run(app, host="0.0.0.0", port=8000)
