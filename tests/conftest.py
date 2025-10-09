import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

@pytest.fixture
def sample_data():
    return {
        "customer_id": "TEST_001",
        "tenure": 12,
        "monthly_charges": 85.50,
        "total_charges": 1026.00,
        "contract_type": "Month-to-month",
        "payment_method": "Electronic check",
        "internet_service": "Fiber optic"
    }

@pytest.fixture
def api_client():
    from fastapi.testclient import TestClient
    from api.main import app
    return TestClient(app)