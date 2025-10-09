import os
from pydantic import BaseSettings

class Settings(BaseSettings):
    API_TITLE: str = "Churn Prediction API"
    API_VERSION: str = "1.0.0"
    MODEL_PATH: str = "models/best_model.pkl"
    SCALER_PATH: str = "models/scaler.pkl"
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"

settings = Settings()