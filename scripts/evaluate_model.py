"""
Script to evaluate the best churn prediction model
"""

import os
import sys
import yaml
import joblib
import pandas as pd

# Add project root to sys.path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model_evaluation import ModelEvaluator


def main():
    print("🚀 Running model evaluation...")

    # Load config file
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'model_config.yaml')
    config_path = os.path.abspath(config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load processed test data
    test_data_path = config['data'].get('processed_test', os.path.join('data', 'processed', 'churn_test.csv'))
    test_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', test_data_path))
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test data not found at {test_data_path}")
    df = pd.read_csv(test_data_path)

    # Drop unnecessary columns if present
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)

    # Prepare features and target
    y_test = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    X_test = pd.get_dummies(df.drop('Churn', axis=1), drop_first=True)

    # Load trained model
    model_path = config['model'].get('best_model_path', os.path.join('models', 'best_tuned_model.pkl'))
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', model_path))
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    model = joblib.load(model_path)

    # Ensure output directory exists
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'docs'))
    os.makedirs(output_dir, exist_ok=True)

    # Evaluate model
    evaluator = ModelEvaluator(config)
    report = evaluator.generate_evaluation_report(model, X_test, y_test, output_dir)

    print("✅ Model evaluation completed successfully.")
    print("\n📊 Summary Report:")
    print(report[:600] + "\n...")  # print first part of report for readability


if __name__ == "__main__":
    main()
