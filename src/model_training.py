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
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Comprehensive model training with MLflow tracking"""
    
    def __init__(self, config: dict):
        self.config = config
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_score = 0
        
        # Setup MLflow
        mlflow.set_tracking_uri(config.get('mlflow', {}).get('tracking_uri', 'mlruns'))
        mlflow.set_experiment(config.get('mlflow', {}).get('experiment_name', 'churn-prediction'))
    
    def prepare_data(self, X: pd.DataFrame, y: np.ndarray) -> Tuple:
        """Prepare train/test splits"""
        test_size = self.config['data']['test_size']
        val_size = self.config['data'].get('validation_size', 0.1)
        stratify = y if self.config['data'].get('stratify', True) else None
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=stratify
        )
        
        # Validation split
        if val_size > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=val_size, random_state=42, 
                stratify=y_train
            )
        else:
            X_val, y_val = None, None
        
        logger.info(f"Train: {len(X_train)}, Val: {len(X_val) if X_val is not None else 0}, Test: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def handle_imbalance(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple:
        """Handle class imbalance"""
        method = self.config['imbalance']['method']
        
        if method == 'smote':
            smote = SMOTE(
                sampling_strategy=self.config['imbalance']['sampling_strategy'],
                random_state=42
            )
            X_train, y_train = smote.fit_resample(X_train, y_train)
            logger.info(f"Applied SMOTE: {len(X_train)} samples")
        
        elif method == 'undersampling':
            from imblearn.under_sampling import RandomUnderSampler
            rus = RandomUnderSampler(random_state=42)
            X_train, y_train = rus.fit_resample(X_train, y_train)
            logger.info(f"Applied undersampling: {len(X_train)} samples")
        
        return X_train, y_train
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                      X_val: pd.DataFrame = None) -> Tuple:
        """Scale numerical features"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        X_val_scaled = None
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
        
        logger.info("Features scaled")
        return X_train_scaled, X_test_scaled, X_val_scaled
    
    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray = None, y_val: np.ndarray = None) -> XGBClassifier:
        """Train XGBoost model"""
        params = self.config['xgboost']
        
        model = XGBClassifier(
            **params,
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False
        )
        
        if X_val is not None:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            model.fit(X_train, y_train)
        
        logger.info("XGBoost training complete")
        return model
    
    def train_lightgbm(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray = None, y_val: np.ndarray = None) -> LGBMClassifier:
        """Train LightGBM model"""
        params = self.config['lightgbm']
        
        model = LGBMClassifier(
            **params,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        if X_val is not None:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='auc'
            )
        else:
            model.fit(X_train, y_train)
        
        logger.info("LightGBM training complete")
        return model
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
        """Train Random Forest model"""
        params = self.config['random_forest']
        
        model = RandomForestClassifier(
            **params,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        logger.info("Random Forest training complete")
        return model
    
    def cross_validate_model(self, model: Any, X: np.ndarray, y: np.ndarray) -> Dict:
        """Perform cross-validation"""
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        scores = {
            'roc_auc': cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1),
            'precision': cross_val_score(model, X, y, cv=cv, scoring='precision', n_jobs=-1),
            'recall': cross_val_score(model, X, y, cv=cv, scoring='recall', n_jobs=-1),
            'f1': cross_val_score(model, X, y, cv=cv, scoring='f1', n_jobs=-1)
        }
        
        cv_results = {
            metric: {
                'mean': scores[metric].mean(),
                'std': scores[metric].std()
            }
            for metric in scores
        }
        
        logger.info(f"Cross-validation complete: ROC-AUC = {cv_results['roc_auc']['mean']:.4f}")
        return cv_results
    
    def train_all_models(self, X: pd.DataFrame, y: np.ndarray) -> Dict:
        """Train all models and track with MLflow"""
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data(X, y)
        
        # Scale features
        X_train_scaled, X_test_scaled, X_val_scaled = self.scale_features(
            X_train, X_test, X_val
        )
        
        # Handle imbalance
        X_train_balanced, y_train_balanced = self.handle_imbalance(
            X_train_scaled, y_train
        )
        
        models = {}
        
        # Train XGBoost
        with mlflow.start_run(run_name="xgboost"):
            mlflow.log_params(self.config['xgboost'])
            
            xgb_model = self.train_xgboost(
                X_train_balanced, y_train_balanced,
                X_val_scaled, y_val
            )
            
            from src.model_evaluation import ModelEvaluator
            evaluator = ModelEvaluator(self.config)
            metrics = evaluator.evaluate_model(xgb_model, X_test_scaled, y_test)
            
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(xgb_model, "model")
            
            models['xgboost'] = {
                'model': xgb_model,
                'metrics': metrics
            }
            
            logger.info(f"XGBoost ROC-AUC: {metrics['roc_auc']:.4f}")
        
        # Train LightGBM
        with mlflow.start_run(run_name="lightgbm"):
            mlflow.log_params(self.config['lightgbm'])
            
            lgbm_model = self.train_lightgbm(
                X_train_balanced, y_train_balanced,
                X_val_scaled, y_val
            )
            
            metrics = evaluator.evaluate_model(lgbm_model, X_test_scaled, y_test)
            
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(lgbm_model, "model")
            
            models['lightgbm'] = {
                'model': lgbm_model,
                'metrics': metrics
            }
            
            logger.info(f"LightGBM ROC-AUC: {metrics['roc_auc']:.4f}")
        
        # Train Random Forest
        with mlflow.start_run(run_name="random_forest"):
            mlflow.log_params(self.config['random_forest'])
            
            rf_model = self.train_random_forest(X_train_balanced, y_train_balanced)
            
            metrics = evaluator.evaluate_model(rf_model, X_test_scaled, y_test)
            
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(rf_model, "model")
            
            models['random_forest'] = {
                'model': rf_model,
                'metrics': metrics
            }
            
            logger.info(f"Random Forest ROC-AUC: {metrics['roc_auc']:.4f}")
        
        # Select best model
        best_model_name = max(models, key=lambda x: models[x]['metrics']['roc_auc'])
        self.best_model = models[best_model_name]['model']
        self.best_score = models[best_model_name]['metrics']['roc_auc']
        
        logger.info(f"Best model: {best_model_name} (ROC-AUC: {self.best_score:.4f})")
        
        return models
    
    def save_model(self, model: Any, filepath: str):
        """Save model"""
        joblib.dump(model, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def save_artifacts(self, output_dir: str = 'models'):
        """Save all artifacts"""
        joblib.dump(self.scaler, f"{output_dir}/scaler.pkl")
        joblib.dump(self.best_model, f"{output_dir}/best_model.pkl")
        logger.info(f"Artifacts saved to {output_dir}")