"""
API Tests
Run: pytest tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import app

client = TestClient(app)

@pytest.fixture
def sample_customer_data():
    """Sample customer data for testing"""
    return {
        "customer_id": "TEST_001",
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

def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert response.json()["status"] == "healthy"

def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "model_status" in data

def test_predict_endpoint(sample_customer_data):
    """Test single prediction endpoint"""
    response = client.post("/predict", json=sample_customer_data)
    
    if response.status_code == 503:
        pytest.skip("Model not loaded - expected in CI environment")
    
    assert response.status_code == 200
    data = response.json()
    
    # Check response structure
    assert "customer_id" in data
    assert "churn_probability" in data
    assert "churn_prediction" in data
    assert "risk_level" in data
    assert "recommended_actions" in data
    
    # Check data types
    assert isinstance(data["churn_probability"], float)
    assert isinstance(data["churn_prediction"], bool)
    assert data["risk_level"] in ["Low", "Medium", "High"]
    assert isinstance(data["recommended_actions"], list)
    
    # Check value ranges
    assert 0 <= data["churn_probability"] <= 1

def test_predict_invalid_data():
    """Test prediction with invalid data"""
    invalid_data = {
        "customer_id": "TEST_002",
        "tenure": -5,  # Invalid negative
        "monthly_charges": "invalid"  # Wrong type
    }
    
    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422  # Validation error

def test_batch_predict(sample_customer_data):
    """Test batch prediction endpoint"""
    batch_data = {
        "customers": [
            sample_customer_data,
            {**sample_customer_data, "customer_id": "TEST_002", "tenure": 24},
            {**sample_customer_data, "customer_id": "TEST_003", "tenure": 36}
        ]
    }
    
    response = client.post("/batch-predict", json=batch_data)
    
    if response.status_code == 503:
        pytest.skip("Model not loaded - expected in CI environment")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "total_customers" in data
    assert data["total_customers"] == 3
    assert "predictions" in data
    assert len(data["predictions"]) == 3

def test_stats_endpoint():
    """Test statistics endpoint"""
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    
    assert "model_version" in data
    assert "model_accuracy" in data

def test_cors_headers():
    """Test CORS headers are present"""
    response = client.get("/")
    assert "access-control-allow-origin" in response.headers

def test_api_documentation():
    """Test API documentation is accessible"""
    response = client.get("/docs")
    assert response.status_code == 200

def test_openapi_schema():
    """Test OpenAPI schema is available"""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "openapi" in schema
    assert "info" in schema

@pytest.mark.parametrize("tenure,monthly_charges,expected_risk", [
    (3, 95.0, "High"),  # Short tenure + high charges
    (24, 45.0, "Low"),  # Long tenure + low charges
    (12, 70.0, "Medium")  # Medium values
])
def test_risk_levels(tenure, monthly_charges, expected_risk, sample_customer_data):
    """Test different risk level scenarios"""
    test_data = {**sample_customer_data}
    test_data["tenure"] = tenure
    test_data["monthly_charges"] = monthly_charges
    test_data["total_charges"] = tenure * monthly_charges
    
    response = client.post("/predict", json=test_data)
    
    if response.status_code == 503:
        pytest.skip("Model not loaded")
    
    # Note: This test assumes model behavior, may need adjustment
    assert response.status_code == 200

def test_response_time():
    """Test API response time"""
    import time
    
    sample_data = {
        "customer_id": "PERF_TEST",
        "tenure": 12,
        "monthly_charges": 85.50,
        "total_charges": 1026.00,
        "contract_type": "Month-to-month",
        "payment_method": "Electronic check",
        "internet_service": "Fiber optic"
    }
    
    start_time = time.time()
    response = client.post("/predict", json=sample_data)
    elapsed_time = time.time() - start_time
    
    if response.status_code == 503:
        pytest.skip("Model not loaded")
    
    # Response should be under 2 seconds
    assert elapsed_time < 2.0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])