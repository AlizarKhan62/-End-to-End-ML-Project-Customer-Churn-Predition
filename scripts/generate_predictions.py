import pandas as pd
import joblib
import sys
sys.path.append('.')

def generate_batch_predictions(input_csv: str, output_csv: str):
    """Generate predictions for batch of customers"""
    
    print(f"📊 Loading data from {input_csv}")
    df = pd.read_csv(input_csv)
    
    print("🤖 Loading model...")
    model = joblib.load('models/best_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    
    print("🔮 Generating predictions...")
    # Add preprocessing here
    X = df  # Preprocess first
    X_scaled = scaler.transform(X)
    
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]
    
    df['churn_prediction'] = predictions
    df['churn_probability'] = probabilities
    df['risk_level'] = pd.cut(probabilities, bins=[0, 0.4, 0.7, 1.0], 
                               labels=['Low', 'Medium', 'High'])
    
    df.to_csv(output_csv, index=False)
    print(f"✅ Predictions saved to {output_csv}")

if __name__ == "__main__":
    generate_batch_predictions(
        'data/raw/customers.csv',
        'data/predictions/predictions.csv'
    )
