import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import pandas as pd
import warnings
import joblib

warnings.filterwarnings("ignore")

def objective(trial):
    """Optuna objective function for XGBoost"""
    
    # Load data
    df = pd.read_csv("data/processed/churn_engineered.csv")

    # Drop ID column if present
    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    # Clean and convert TotalCharges column safely
    df["TotalCharges"] = (
        df["TotalCharges"]
        .astype(str)
        .str.strip()
        .replace("", "0")
    )
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Separate features and target
    X = df.drop("Churn", axis=1)
    y = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)

    # One-hot encode categorical columns
    X = pd.get_dummies(X, drop_first=True)

    # Split data
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Suggest hyperparameters
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': 42,
        'use_label_encoder': False,
        'eval_metric': 'logloss'
    }

    # Train model
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred_prob = model.predict_proba(X_valid)[:, 1]
    score = roc_auc_score(y_valid, y_pred_prob)

    return score


def run_tuning():
    print("🔧 Starting Hyperparameter Tuning...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=25, timeout=1800)

    print(f"\n✅ Best Parameters: {study.best_params}")
    print(f"✅ Best ROC-AUC: {study.best_value:.4f}")

    # Reload and preprocess data for final training
    df = pd.read_csv("data/processed/churn_engineered.csv")
    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)

    df["TotalCharges"] = (
        df["TotalCharges"]
        .astype(str)
        .str.strip()
        .replace("", "0")
    )
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    X = pd.get_dummies(df.drop("Churn", axis=1), drop_first=True)
    y = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train best model again
    best_model = XGBClassifier(**study.best_params, random_state=42, use_label_encoder=False, eval_metric="logloss")
    best_model.fit(X_train, y_train)

    # Save best tuned model
    joblib.dump(best_model, "models/best_tuned_model.pkl")
    print("\n💾 Model saved as: models/best_tuned_model.pkl")


if __name__ == "__main__":
    run_tuning()
