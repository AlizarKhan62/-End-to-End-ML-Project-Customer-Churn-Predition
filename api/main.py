"""
FastAPI Application for Churn Prediction
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import os

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Production-ready ML API for predicting customer churn",
    version="1.0.0"
)

# ✅ Enable full CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load artifacts
MODEL_PATH = "models/best_model.pkl"
SCALER_PATH = "models/scaler.pkl"
FEATURES_PATH = "models/feature_names.pkl"

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    logger.info("✅ Model, Scaler & Feature Names loaded")
except Exception as e:
    logger.error(f"Model load error: {e}")
    model, scaler, feature_names = None, None, None

# ------------------------------- Schemas -------------------------------
class CustomerData(BaseModel):
    customer_id: str
    tenure: int
    monthly_charges: float
    total_charges: float
    contract_type: str
    payment_method: str
    internet_service: str
    online_security: Optional[str]
    tech_support: Optional[str]
    device_protection: Optional[str]
    streaming_tv: Optional[str]
    streaming_movies: Optional[str]

class PredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    churn_prediction: bool
    risk_level: str
    confidence: float
    timestamp: str
    recommended_actions: List[str]

class BatchPredictionRequest(BaseModel):
    customers: List[CustomerData]

# -------------------------- Helper Functions --------------------------
def preprocess_input(data: CustomerData) -> np.ndarray:
    df = pd.DataFrame([data.dict()])
    df['avg_monthly_charges'] = df['total_charges'] / (df['tenure'] + 1)
    df['charge_to_tenure_ratio'] = df['monthly_charges'] / (df['tenure'] + 1)
    df = pd.get_dummies(df, drop_first=True)

    # 👇 Align with training columns
    if feature_names:
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_names]
    else:
        logger.warning("Feature names not loaded. Using raw columns.")

    if scaler:
        df_scaled = scaler.transform(df)
    else:
        df_scaled = df.values

    return df_scaled

def get_risk_level(probability: float) -> str:
    if probability >= 0.7:
        return "High"
    elif probability >= 0.4:
        return "Medium"
    return "Low"

def get_recommendations(prob: float, data: CustomerData) -> List[str]:
    r = []
    if prob >= 0.7:
        r += ["🚨 Immediate intervention", "💰 Offer loyalty discount", "📞 Retention call"]
    elif prob >= 0.4:
        r += ["📊 Monitor satisfaction", "🎁 Offer perks"]
    else:
        r += ["✅ Relationship is healthy"]
    if data.contract_type == "Month-to-month":
        r.append("📝 Suggest long-term contract")
    return r

# -------------------------- API Endpoints --------------------------
@app.get("/")
async def root():
    return {"message": "Customer Churn Prediction API", "status": "healthy"}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": "loaded" if model else "not_loaded",
        "scaler": "loaded" if scaler else "not_loaded",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(data: CustomerData):
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        X = preprocess_input(data)
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0][1]
        response = PredictionResponse(
            customer_id=data.customer_id,
            churn_probability=float(round(prob, 4)),
            churn_prediction=bool(pred),
            risk_level=get_risk_level(prob),
            confidence=float(round(max(model.predict_proba(X)[0]), 4)),
            timestamp=datetime.now().isoformat(),
            recommended_actions=get_recommendations(prob, data)
        )
        return response
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/batch-predict")
async def batch_predict(req: BatchPredictionRequest):
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        results = []
        for c in req.customers:
            X = preprocess_input(c)
            pred = model.predict(X)[0]
            prob = model.predict_proba(X)[0][1]
            results.append({
                "customer_id": c.customer_id,
                "probability": float(round(prob, 4)),
                "prediction": bool(pred),
                "risk_level": get_risk_level(prob)
            })
        return {"total": len(results), "predictions": results}
    except Exception as e:
        logger.error(f"Batch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
