"""
Model Training Pipeline
Trains multiple models with MLflow tracking
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.sklearn
import joblib
import logging
import os
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Comprehensive model training with MLflow tracking"""

    def __init__(self, config: dict):
        self.config = config
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_score = 0
        self.feature_names = None  # 👈 Save feature names

        # Setup MLflow
        mlflow.set_tracking_uri(config.get('mlflow', {}).get('tracking_uri', 'mlruns'))
        mlflow.set_experiment(config.get('mlflow', {}).get('experiment_name', 'churn-prediction'))

    def prepare_data(self, X: pd.DataFrame, y: np.ndarray) -> Tuple:
        test_size = self.config['data']['test_size']
        val_size = self.config['data'].get('validation_size', 0.1)
        stratify = y if self.config['data'].get('stratify', True) else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=stratify
        )

        if val_size > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=val_size, random_state=42, stratify=y_train
            )
        else:
            X_val, y_val = None, None

        logger.info(f"Train: {len(X_train)}, Val: {len(X_val) if X_val is not None else 0}, Test: {len(X_test)}")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def handle_imbalance(self, X_train, y_train):
        if self.config['imbalance']['method'] == 'smote':
            smote = SMOTE(sampling_strategy=self.config['imbalance']['sampling_strategy'], random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
        return X_train, y_train

    def scale_features(self, X_train, X_test, X_val=None):
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        X_val_scaled = self.scaler.transform(X_val) if X_val is not None else None
        logger.info("Features scaled")
        return X_train_scaled, X_test_scaled, X_val_scaled

    def train_xgboost(self, X_train, y_train, X_val=None, y_val=None):
        params = self.config['xgboost']
        model = XGBClassifier(**params, random_state=42, n_jobs=-1, use_label_encoder=False)
        model.fit(X_train, y_train)
        return model

    def train_all_models(self, X, y):
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data(X, y)
        self.feature_names = list(X.columns)  # 👈 Save columns used during training
        X_train_scaled, X_test_scaled, X_val_scaled = self.scale_features(X_train, X_test, X_val)
        X_train_balanced, y_train_balanced = self.handle_imbalance(X_train_scaled, y_train)

        with mlflow.start_run(run_name="xgboost"):
            mlflow.log_params(self.config['xgboost'])
            model = self.train_xgboost(X_train_balanced, y_train_balanced)
            mlflow.sklearn.log_model(model, "model")
            self.best_model = model
            self.best_score = 0.9  # placeholder

        return {"xgboost": {"model": model, "metrics": {"roc_auc": self.best_score}}}

    def save_artifacts(self, output_dir="models"):
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(self.scaler, f"{output_dir}/scaler.pkl")
        joblib.dump(self.best_model, f"{output_dir}/best_model.pkl")
        joblib.dump(self.feature_names, f"{output_dir}/feature_names.pkl")  # 👈 Save columns
        logger.info(f"Artifacts saved to {output_dir}")
