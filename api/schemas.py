from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class CustomerBase(BaseModel):
    customer_id: str
    tenure: int = Field(ge=0)
    monthly_charges: float = Field(ge=0)
    total_charges: float = Field(ge=0)

class PredictionInput(CustomerBase):
    contract_type: str
    payment_method: str
    internet_service: str
    online_security: Optional[str] = None
    tech_support: Optional[str] = None

class PredictionOutput(BaseModel):
    customer_id: str
    churn_probability: float
    churn_prediction: bool
    risk_level: str
    confidence: float
    timestamp: datetime
    recommended_actions: List[str]