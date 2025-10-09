"""
FastAPI Application for Churn Prediction
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import os
from dotenv import load_dotenv

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Production-ready ML API for predicting customer churn",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models at startup
MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "models/scaler.pkl")

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {e}")
    model = None
    scaler = None

# Pydantic schemas
class CustomerData(BaseModel):
    """Customer data schema"""
    customer_id: str = Field(..., description="Unique customer identifier")
    tenure: int = Field(..., ge=0, description="Months with company")
    monthly_charges: float = Field(..., ge=0, description="Monthly charges")
    total_charges: float = Field(..., ge=0, description="Total charges")
    contract_type: str = Field(..., description="Contract type")
    payment_method: str = Field(..., description="Payment method")
    internet_service: str = Field(..., description="Internet service type")
    online_security: Optional[str] = Field(None, description="Online security")
    tech_support: Optional[str] = Field(None, description="Tech support")
    device_protection: Optional[str] = Field(None, description="Device protection")
    streaming_tv: Optional[str] = Field(None, description="Streaming TV")
    streaming_movies: Optional[str] = Field(None, description="Streaming movies")
    
    class Config:
        schema_extra = {
            "example": {
                "customer_id": "CUST_001",
                "tenure": 12,
                "monthly_charges": 85.50,
                "total_charges": 1026.00,
                "contract_type": "Month-to-month",
                "payment_method": "Electronic check",
                "internet_service": "Fiber optic",
                "online_security": "No",
                "tech_support": "No",
                "device_protection": "No",
                "streaming_tv": "Yes",
                "streaming_movies": "Yes"
            }
        }

class PredictionResponse(BaseModel):
    """Prediction response schema"""
    customer_id: str
    churn_probability: float
    churn_prediction: bool
    risk_level: str
    confidence: float
    timestamp: str
    recommended_actions: List[str]

class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    customers: List[CustomerData]

# Helper functions
def preprocess_input(data: CustomerData) -> np.ndarray:
    """Preprocess input data"""
    # Convert to DataFrame
    df = pd.DataFrame([data.dict()])
    
    # Feature engineering (simplified)
    df['avg_monthly_charges'] = df['total_charges'] / (df['tenure'] + 1)
    df['charge_to_tenure_ratio'] = df['monthly_charges'] / (df['tenure'] + 1)
    
    # Encode categorical variables (simplified - in production use saved encoders)
    categorical_cols = ['contract_type', 'payment_method', 'internet_service', 
                       'online_security', 'tech_support', 'device_protection',
                       'streaming_tv', 'streaming_movies']
    
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category').cat.codes
    
    # Remove customer_id
    df = df.drop('customer_id', axis=1, errors='ignore')
    
    # Scale features
    if scaler:
        df_scaled = scaler.transform(df)
    else:
        df_scaled = df.values
    
    return df_scaled

def get_risk_level(probability: float) -> str:
    """Determine risk level"""
    if probability >= 0.7:
        return "High"
    elif probability >= 0.4:
        return "Medium"
    else:
        return "Low"

def get_recommendations(probability: float, data: CustomerData) -> List[str]:
    """Get personalized recommendations"""
    recommendations = []
    
    if probability >= 0.7:
        recommendations.append("🚨 Immediate intervention required")
        recommendations.append("💰 Offer loyalty discount (15-20%)")
        recommendations.append("📞 Schedule retention call within 48 hours")
    
    if data.contract_type == "Month-to-month":
        recommendations.append("📝 Incentivize annual contract upgrade")
    
    if data.tenure < 12:
        recommendations.append("🎁 New customer retention program")
    
    if data.monthly_charges > 80:
        recommendations.append("💳 Review pricing plan optimization")
    
    if data.tech_support == "No":
        recommendations.append("🛠️ Offer complimentary tech support trial")
    
    if not recommendations:
        recommendations.append("✅ Customer relationship is healthy")
        recommendations.append("📧 Send satisfaction survey")
    
    return recommendations

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Customer Churn Prediction API",
        "version": "1.0.0",
        "status": "healthy",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/batch-predict",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_status = "loaded" if model is not None else "not_loaded"
    scaler_status = "loaded" if scaler is not None else "not_loaded"
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_status": model_status,
        "scaler_status": scaler_status,
        "version": "1.0.0"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(data: CustomerData):
    """Predict churn for single customer"""
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Preprocess
        X = preprocess_input(data)
        
        # Predict
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][1]
        
        # Get risk level and recommendations
        risk_level = get_risk_level(probability)
        recommendations = get_recommendations(probability, data)
        
        response = PredictionResponse(
            customer_id=data.customer_id,
            churn_probability=round(float(probability), 4),
            churn_prediction=bool(prediction),
            risk_level=risk_level,
            confidence=round(float(max(model.predict_proba(X)[0])), 4),
            timestamp=datetime.now().isoformat(),
            recommended_actions=recommendations
        )
        
        logger.info(f"Prediction for {data.customer_id}: {probability:.4f}")
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/batch-predict")
async def batch_predict(request: BatchPredictionRequest):
    """Batch prediction for multiple customers"""
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        predictions = []
        
        for customer in request.customers:
            X = preprocess_input(customer)
            prediction = model.predict(X)[0]
            probability = model.predict_proba(X)[0][1]
            
            predictions.append({
                "customer_id": customer.customer_id,
                "churn_probability": round(float(probability), 4),
                "churn_prediction": bool(prediction),
                "risk_level": get_risk_level(probability)
            })
        
        logger.info(f"Batch prediction completed: {len(predictions)} customers")
        
        return {
            "total_customers": len(predictions),
            "high_risk_count": sum(1 for p in predictions if p['risk_level'] == 'High'),
            "predictions": predictions,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/stats")
async def get_statistics():
    """Get model statistics"""
    try:
        # In production, load from database
        return {
            "model_version": "1.0.0",
            "total_predictions": 0,
            "average_churn_rate": 0.26,
            "model_accuracy": 0.92,
            "last_retrained": "2025-10-01",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)