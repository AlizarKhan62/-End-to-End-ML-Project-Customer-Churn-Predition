"""
Power BI Connector Script
Use this to connect Power BI to your API
"""

import requests
import pandas as pd
from typing import List, Dict
import json

class PowerBIConnector:
    """Connector for Power BI Dashboard"""
    
    def __init__(self, api_url: str):
        self.api_url = api_url.rstrip('/')
        self.test_connection()
    
    def test_connection(self) -> bool:
        """Test API connection"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=10)
            if response.status_code == 200:
                print("✅ API connection successful")
                return True
            else:
                print(f"❌ API connection failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    def get_batch_predictions(self, customers_df: pd.DataFrame) -> pd.DataFrame:
        """Get predictions for multiple customers"""
        
        # Convert DataFrame to API format
        customers_list = customers_df.to_dict('records')
        
        payload = {"customers": customers_list}
        
        try:
            response = requests.post(
                f"{self.api_url}/batch-predict",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                predictions_df = pd.DataFrame(result['predictions'])
                return predictions_df
            else:
                print(f"Error: {response.status_code}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error getting predictions: {e}")
            return pd.DataFrame()
    
    def get_single_prediction(self, customer_data: dict) -> dict:
        """Get prediction for single customer"""
        try:
            response = requests.post(
                f"{self.api_url}/predict",
                json=customer_data,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Status code: {response.status_code}"}
                
        except Exception as e:
            return {"error": str(e)}
    
    def export_for_powerbi(self, input_csv: str, output_csv: str):
        """Export predictions to CSV for Power BI"""
        
        # Read input data
        df = pd.read_csv(input_csv)
        
        # Get predictions
        predictions_df = self.get_batch_predictions(df)
        
        # Merge with original data
        result_df = df.merge(
            predictions_df,
            on='customer_id',
            how='left'
        )
        
        # Save to CSV
        result_df.to_csv(output_csv, index=False)
        print(f"✅ Predictions exported to {output_csv}")
        
        return result_df


# ============================================================================
# POWER BI M CODE (Power Query)
# Copy this into Power BI Advanced Editor
# ============================================================================

POWERBI_M_CODE = '''
let
    // API Configuration
    ApiUrl = "YOUR_API_URL_HERE",
    
    // Load your customer data
    Source = Csv.Document(File.Contents("C:\\path\\to\\your\\customers.csv"),[Delimiter=",", Columns=auto, Encoding=65001, QuoteStyle=QuoteStyle.None]),
    
    // Promote headers
    PromotedHeaders = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    
    // Change types
    TypedData = Table.TransformColumnTypes(PromotedHeaders,{
        {"customer_id", type text}, 
        {"tenure", Int64.Type}, 
        {"monthly_charges", type number}, 
        {"total_charges", type number},
        {"contract_type", type text},
        {"payment_method", type text},
        {"internet_service", type text}
    }),
    
    // Function to call API
    GetPrediction = (row as record) =>
        let
            JsonBody = Json.FromValue(row),
            Response = Web.Contents(
                ApiUrl & "/predict",
                [
                    Headers=[#"Content-Type"="application/json"],
                    Content=JsonBody
                ]
            ),
            JsonResponse = Json.Document(Response)
        in
            JsonResponse,
    
    // Add predictions column
    WithPredictions = Table.AddColumn(
        TypedData,
        "Prediction",
        each GetPrediction(_)
    ),
    
    // Expand prediction results
    ExpandedPredictions = Table.ExpandRecordColumn(
        WithPredictions,
        "Prediction",
        {"churn_probability", "churn_prediction", "risk_level", "recommended_actions"},
        {"churn_probability", "churn_prediction", "risk_level", "recommended_actions"}
    )
in
    ExpandedPredictions
'''

# ============================================================================
# POWER BI PYTHON SCRIPT (Alternative Method)
# Use this in Power BI Python Script Visual
# ============================================================================

POWERBI_PYTHON_SCRIPT = '''
import requests
import pandas as pd

# API URL (replace with your actual URL)
API_URL = "YOUR_API_URL_HERE"

# dataset is the input DataFrame from Power BI
customers = dataset.to_dict('records')

# Call API
response = requests.post(
    f"{API_URL}/batch-predict",
    json={"customers": customers},
    timeout=60
)

# Get predictions
if response.status_code == 200:
    predictions = pd.DataFrame(response.json()['predictions'])
    
    # Merge with original data
    result = dataset.merge(predictions, on='customer_id', how='left')
    
    # This result is available in Power BI
    print(result)
else:
    print(f"Error: {response.status_code}")
'''

# ============================================================================
# DAX MEASURES FOR POWER BI
# ============================================================================

DAX_MEASURES = '''
-- Total Customers
Total Customers = COUNT(Predictions[customer_id])

-- Churn Rate
Churn Rate = 
DIVIDE(
    COUNTROWS(FILTER(Predictions, Predictions[churn_prediction] = TRUE)),
    COUNTROWS(Predictions),
    0
) * 100

-- High Risk Customers
High Risk Count = 
COUNTROWS(FILTER(Predictions, Predictions[risk_level] = "High"))

-- Revenue at Risk
Revenue at Risk = 
SUMX(
    FILTER(Predictions, Predictions[risk_level] = "High"),
    Predictions[monthly_charges] * 12
)

-- Average Churn Probability
Avg Churn Probability = AVERAGE(Predictions[churn_probability])

-- Customers Saved (assuming 60% retention success)
Customers Saved = [High Risk Count] * 0.6

-- ROI Calculation
Retention ROI = 
VAR CampaignCost = [High Risk Count] * 50
VAR RevenueSaved = [Customers Saved] * AVERAGE(Predictions[monthly_charges]) * 12
VAR ROI = DIVIDE(RevenueSaved - CampaignCost, CampaignCost, 0) * 100
RETURN ROI

-- Risk Distribution
Risk Distribution = 
SWITCH(
    TRUE(),
    Predictions[churn_probability] >= 0.7, "High Risk",
    Predictions[churn_probability] >= 0.4, "Medium Risk",
    "Low Risk"
)

-- Tenure Category
Tenure Category = 
SWITCH(
    TRUE(),
    Predictions[tenure] <= 12, "0-1 Year",
    Predictions[tenure] <= 24, "1-2 Years",
    Predictions[tenure] <= 48, "2-4 Years",
    Predictions[tenure] <= 72, "4-6 Years",
    "6+ Years"
)

-- Monthly Charges Category
Price Category = 
SWITCH(
    TRUE(),
    Predictions[monthly_charges] < 30, "Budget",
    Predictions[monthly_charges] < 60, "Standard",
    Predictions[monthly_charges] < 90, "Premium",
    "Elite"
)
'''

def create_powerbi_guide():
    """Create Power BI integration guide"""
    
    guide = """
# Power BI Integration Guide

## Method 1: Direct API Integration (Recommended)

### Step 1: Get Data
1. Open Power BI Desktop
2. Click "Get Data" → "Web"
3. Enter your API URL: `http://your-api-url.com`
4. Click "Advanced" and add headers if needed

### Step 2: Transform Data
1. Click "Transform Data"
2. Add Custom Column → "Get Predictions"
3. Use the M code provided above

### Step 3: Create Visuals
- Use the DAX measures provided
- Create KPI cards for key metrics
- Add slicers for filtering

## Method 2: Python Script

### Step 1: Enable Python in Power BI
1. File → Options → Python scripting
2. Set Python installation directory

### Step 2: Run Python Script
1. Get Data → More → Python script
2. Paste the Python script provided above
3. Replace API_URL with your URL

## Method 3: CSV Export + Import

### Step 1: Export Predictions
```python
from dashboard.powerbi_connector import PowerBIConnector

connector = PowerBIConnector("http://your-api-url.com")
connector.export_for_powerbi(
    input_csv="data/customers.csv",
    output_csv="data/predictions.csv"
)
```

### Step 2: Import to Power BI
1. Get Data → Text/CSV
2. Select predictions.csv
3. Set up auto-refresh schedule

## Recommended Dashboard Structure

### Page 1: Executive Overview
- Total Customers (KPI Card)
- Churn Rate % (KPI Card)
- Revenue at Risk (KPI Card)
- High Risk Count (KPI Card)
- Churn Trend (Line Chart)
- Risk Distribution (Donut Chart)

### Page 2: Customer Analysis
- Risk by Tenure (Column Chart)
- Risk by Contract Type (Bar Chart)
- Risk by Price Category (Matrix)
- Customer Details Table (Table)

### Page 3: Action Dashboard
- High Risk Customers (Table with filters)
- Recommended Actions (Card)
- ROI Calculator (Measure)
- What-If Analysis (Slider)

## Refresh Options

### Option A: Manual Refresh
- Refresh button in Power BI

### Option B: Scheduled Refresh
- Publish to Power BI Service
- Set up schedule (hourly/daily)

### Option C: Real-time with Streaming Dataset
- Create streaming dataset
- Push data via API

## Tips for Best Performance

1. **Limit API Calls**: Cache predictions, refresh periodically
2. **Use DirectQuery**: For real-time data
3. **Optimize Measures**: Use variables in DAX
4. **Filter Early**: Apply filters before API calls
5. **Monitor Costs**: Track API usage

## Troubleshooting

### Error: "Unable to connect"
- Check API is running
- Verify URL is correct
- Check firewall/CORS settings

### Error: "Timeout"
- Increase timeout in M code
- Use batch predictions
- Implement pagination

### Error: "Invalid JSON"
- Validate API response format
- Check data types
- Handle null values
"""
    
    return guide

if __name__ == "__main__":
    # Example usage
    print("Power BI Connector Examples")
    print("="*60)
    
    print("\n1. M Code for Power Query:")
    print(POWERBI_M_CODE)
    
    print("\n2. Python Script for Power BI:")
    print(POWERBI_PYTHON_SCRIPT)
    
    print("\n3. DAX Measures:")
    print(DAX_MEASURES)
    
    print("\n4. Complete Integration Guide:")
    guide = create_powerbi_guide()
    
    # Save guide
    with open('docs/powerbi_integration_guide.md', 'w') as f:
        f.write(guide)
    
    print("✅ Guide saved to docs/powerbi_integration_guide.md")