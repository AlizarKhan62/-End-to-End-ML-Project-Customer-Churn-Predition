"""
Google Colab Deployment Script
Run this in Google Colab to deploy your API
"""

import os
import subprocess
import time

def setup_colab_environment():
    """Setup Google Colab environment"""
    print("🚀 Setting up Google Colab Environment...")
    
    # Install required packages
    packages = [
        "fastapi",
        "uvicorn[standard]",
        "pyngrok",
        "python-dotenv"
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        subprocess.run(["pip", "install", "-q", package], check=True)
    
    print("✅ Packages installed")

def clone_repository(repo_url: str = None):
    """Clone GitHub repository"""
    if repo_url:
        print(f"📥 Cloning repository: {repo_url}")
        subprocess.run(["git", "clone", repo_url], check=True)
        
        # Change to repo directory
        repo_name = repo_url.split("/")[-1].replace(".git", "")
        os.chdir(repo_name)
        print(f"✅ Changed to directory: {repo_name}")
    else:
        print("⚠️  No repository URL provided. Using current directory.")

def setup_ngrok(auth_token: str):
    """Setup ngrok for public URL"""
    print("🌐 Setting up ngrok...")
    
    from pyngrok import ngrok, conf
    
    # Set auth token
    conf.get_default().auth_token = auth_token
    
    print("✅ ngrok configured")
    return ngrok

def start_api_server(port: int = 8000):
    """Start FastAPI server"""
    print(f"🚀 Starting API server on port {port}...")
    
    # Start uvicorn in background
    import threading
    import uvicorn
    
    config = uvicorn.Config(
        "api.main:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
    server = uvicorn.Server(config)
    
    thread = threading.Thread(target=server.run)
    thread.daemon = True
    thread.start()
    
    # Wait for server to start
    time.sleep(3)
    print("✅ API server started")

def deploy_to_colab(repo_url: str = None, ngrok_token: str = None):
    """Complete deployment pipeline for Colab"""
    
    print("="*60)
    print("🎯 GOOGLE COLAB DEPLOYMENT")
    print("="*60)
    
    # Step 1: Setup environment
    setup_colab_environment()
    
    # Step 2: Clone repository (if provided)
    if repo_url:
        clone_repository(repo_url)
    
    # Step 3: Install project dependencies
    print("\n📦 Installing project dependencies...")
    if os.path.exists("requirements.txt"):
        subprocess.run(["pip", "install", "-q", "-r", "requirements.txt"], check=True)
        print("✅ Dependencies installed")
    
    # Step 4: Start API server
    start_api_server()
    
    # Step 5: Setup ngrok (if token provided)
    if ngrok_token:
        from pyngrok import ngrok
        ngrok.set_auth_token(ngrok_token)
        
        # Create tunnel
        public_url = ngrok.connect(8000)
        print(f"\n🌐 Public URL: {public_url}")
        print(f"📚 API Docs: {public_url}/docs")
        print(f"🏥 Health Check: {public_url}/health")
    else:
        print("\n⚠️  No ngrok token provided. API accessible locally only.")
        print("🔗 Local URL: http://localhost:8000")
    
    print("\n" + "="*60)
    print("✅ DEPLOYMENT COMPLETE!")
    print("="*60)
    print("\n📝 Next Steps:")
    print("1. Test the API using the URLs above")
    print("2. Keep this Colab notebook running")
    print("3. Connect Power BI to the public URL")
    
    # Keep running
    try:
        print("\n⏳ Server running... (Press Ctrl+C to stop)")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped")

# ============================================================================
# USAGE EXAMPLES FOR GOOGLE COLAB
# ============================================================================

"""
# EXAMPLE 1: Deploy from GitHub repository
# Copy this into a Colab cell:

!git clone https://github.com/yourusername/churn-prediction.git
%cd churn-prediction
!pip install -r requirements.txt

# Get ngrok token from: https://dashboard.ngrok.com/get-started/your-authtoken
NGROK_TOKEN = "your_ngrok_token_here"

from scripts.deploy_colab import deploy_to_colab
deploy_to_colab(ngrok_token=NGROK_TOKEN)
"""

"""
# EXAMPLE 2: Quick deployment (current directory)
# If you've already uploaded files to Colab:

NGROK_TOKEN = "your_ngrok_token_here"

from scripts.deploy_colab import deploy_to_colab
deploy_to_colab(ngrok_token=NGROK_TOKEN)
"""

"""
# EXAMPLE 3: Simple local deployment (no public URL)

from scripts.deploy_colab import setup_colab_environment, start_api_server

setup_colab_environment()
start_api_server()

print("API running at: http://localhost:8000")
print("Access docs at: http://localhost:8000/docs")
"""

# ============================================================================
# COMPLETE COLAB NOTEBOOK CODE
# ============================================================================

COLAB_NOTEBOOK_CODE = """
# Customer Churn Prediction API - Google Colab Deployment

## Step 1: Setup Environment

```python
# Install dependencies
!pip install fastapi uvicorn[standard] pyngrok python-dotenv -q
!pip install scikit-learn xgboost lightgbm pandas numpy joblib -q

# Clone your repository
!git clone https://github.com/yourusername/churn-prediction.git
%cd churn-prediction

# Install project requirements
!pip install -r requirements.txt -q
```

## Step 2: Configure ngrok

```python
# Get your token from: https://dashboard.ngrok.com/get-started/your-authtoken
from pyngrok import ngrok, conf

NGROK_TOKEN = "YOUR_NGROK_TOKEN_HERE"
ngrok.set_auth_token(NGROK_TOKEN)
```

## Step 3: Start API Server

```python
import threading
import uvicorn
from pyngrok import ngrok

# Start FastAPI in background thread
def run_server():
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, log_level="info")

thread = threading.Thread(target=run_server, daemon=True)
thread.start()

# Wait for server to start
import time
time.sleep(5)

# Create public tunnel
public_url = ngrok.connect(8000)
print(f"🌐 Public URL: {public_url}")
print(f"📚 API Documentation: {public_url}/docs")
print(f"🏥 Health Check: {public_url}/health")
```

## Step 4: Test API

```python
import requests

# Test health endpoint
response = requests.get(f"{public_url}/health")
print(response.json())

# Test prediction
test_data = {
    "customer_id": "TEST_001",
    "tenure": 12,
    "monthly_charges": 85.50,
    "total_charges": 1026.00,
    "contract_type": "Month-to-month",
    "payment_method": "Electronic check",
    "internet_service": "Fiber optic",
    "online_security": "No",
    "tech_support": "No"
}

response = requests.post(f"{public_url}/predict", json=test_data)
print(response.json())
```

## Step 5: Keep Server Running

```python
# Keep this cell running to maintain the server
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Server stopped")
```

## Important Notes:
1. Keep the Colab notebook running for the API to stay active
2. Free ngrok URLs change each time you restart
3. Colab has runtime limits (12 hours for free tier)
4. For production, use proper cloud deployment (GCP, AWS, Azure)
"""

if __name__ == "__main__":
    # Interactive mode
    print("🚀 Google Colab Deployment Script")
    print("="*60)
    print("\nThis script helps you deploy the API to Google Colab")
    print("\nOptions:")
    print("1. Deploy from GitHub repository")
    print("2. Deploy from current directory")
    print("3. Show complete Colab notebook code")
    
    choice = input("\nEnter choice (1-3): ")
    
    if choice == "1":
        repo_url = input("Enter GitHub repository URL: ")
        ngrok_token = input("Enter ngrok auth token: ")
        deploy_to_colab(repo_url=repo_url, ngrok_token=ngrok_token)
    
    elif choice == "2":
        ngrok_token = input("Enter ngrok auth token (or press Enter to skip): ")
        token = ngrok_token if ngrok_token else None
        deploy_to_colab(ngrok_token=token)
    
    elif choice == "3":
        print("\n" + "="*60)
        print("COMPLETE COLAB NOTEBOOK CODE")
        print("="*60)
        print(COLAB_NOTEBOOK_CODE)
    
    else:
        print("Invalid choice!")