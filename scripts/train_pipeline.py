"""
Complete Training Pipeline
Run: python scripts/train_pipeline.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import logging
from pathlib import Path
from datetime import datetime
import json
import mlflow  # ✅ Added MLflow import

from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer
from src.model_evaluation import ModelEvaluator
from src.explainability import ModelExplainer

# ============================================================
# Logging setup
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/model_config.yaml") -> dict:
    """Load configuration"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Configuration loaded from {config_path}")
    return config


def main():
    """Main training pipeline"""
    logger.info("=" * 60)
    logger.info("CUSTOMER CHURN PREDICTION - TRAINING PIPELINE")
    logger.info("=" * 60)

    # ============================================================
    # Configure MLflow tracking (force local mode)
    # ============================================================
    tracking_uri = "file:./mlruns"  # ✅ Force MLflow to use local directory
    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"📁 MLflow Tracking URI set to: {tracking_uri}")

    # Ensure experiment exists (create if not)
    try:
        mlflow.set_experiment("churn-prediction")
    except Exception as e:
        logger.warning(f"Could not set experiment: {e}")

    # ============================================================
    # Load configuration
    # ============================================================
    config = load_config()

    # Create output directories
    Path("models/model_registry").mkdir(parents=True, exist_ok=True)
    Path("models/artifacts").mkdir(parents=True, exist_ok=True)
    Path("docs/images").mkdir(parents=True, exist_ok=True)

    # ============================================================
    # Step 1: Data Preprocessing
    # ============================================================
    logger.info("\n📊 Step 1: Data Preprocessing")
    logger.info("-" * 60)

    preprocessor = DataPreprocessor(config["data"])
    data_path = "data/raw/Telco-Customer-Churn.csv"

    if not os.path.exists(data_path):
        logger.error(f"❌ Data file not found: {data_path}")
        logger.info("Please add your data to data/raw/Telco-Customer-Churn.csv")
        return

    df = preprocessor.load_data(data_path)
    X, y = preprocessor.preprocess(df, target_col="Churn", fit=True)
    preprocessor.save_encoders("models/artifacts/label_encoders.pkl")

    # ============================================================
    # Step 2: Feature Engineering
    # ============================================================
    logger.info("\n🔧 Step 2: Feature Engineering")
    logger.info("-" * 60)

    engineer = FeatureEngineer(config["features"])
    X_engineered = engineer.engineer_features(X, fit=True)
    logger.info(f"Features created: {X_engineered.shape[1]}")

    # ============================================================
    # Step 3: Model Training
    # ============================================================
    logger.info("\n🎓 Step 3: Model Training")
    logger.info("-" * 60)

    trainer = ModelTrainer(config)
    models = trainer.train_all_models(X_engineered, y)

    best_model_name = max(models, key=lambda x: models[x]["metrics"]["roc_auc"])
    best_model = models[best_model_name]["model"]

    logger.info(f"\n🏆 Best Model: {best_model_name}")
    metrics = models[best_model_name]["metrics"]
    logger.info(f"   ROC-AUC: {metrics.get('roc_auc', 0):.4f}")

    trainer.save_artifacts("models/artifacts")

    # ============================================================
    # Step 4: Model Evaluation
    # ============================================================
    logger.info("\n📊 Step 4: Model Evaluation")
    logger.info("-" * 60)

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X_engineered, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_scaled = trainer.scaler.transform(X_train)
    X_test_scaled = trainer.scaler.transform(X_test)

    evaluator = ModelEvaluator(config)
    report = evaluator.generate_evaluation_report(
        best_model, X_test_scaled, y_test, output_dir="docs/images"
    )

    # ============================================================
    # Step 5: Model Explainability
    # ============================================================
    if config["explainability"].get("enable_shap", True):
        logger.info("\n🔍 Step 5: Model Explainability")
        logger.info("-" * 60)

        explainer = ModelExplainer(
            best_model,
            X_train_scaled[:100],
            feature_names=X_engineered.columns.tolist(),
        )
        explainer.generate_explanation_report(
            X_test_scaled, output_dir="docs/images"
        )

    # ============================================================
    # Step 6: Save Metadata
    # ============================================================
    logger.info("\n💾 Step 6: Saving Metadata")
    logger.info("-" * 60)

    metadata = {
        "model_version": config["model"]["version"],
        "model_type": best_model_name,
        "training_date": datetime.now().isoformat(),
        "data_shape": {
            "n_samples": len(df),
            "n_features": X_engineered.shape[1],
        },
        "performance": metrics,
        "feature_names": X_engineered.columns.tolist(),
        "config": config,
    }

    with open("models/model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("✅ Metadata saved to models/model_metadata.json")

    # ============================================================
    # Summary (Safe logging)
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("✨ TRAINING COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"\n📊 Best Model: {best_model_name}")
    logger.info(f"   ROC-AUC: {metrics.get('roc_auc', 0):.4f}")

    # ✅ Safely print only available metrics
    if "precision" in metrics:
        logger.info(f"   Precision: {metrics['precision']:.4f}")
    if "recall" in metrics:
        logger.info(f"   Recall: {metrics['recall']:.4f}")
    if "f1" in metrics:
        logger.info(f"   F1-Score: {metrics['f1']:.4f}")
    if "accuracy" in metrics:
        logger.info(f"   Accuracy: {metrics['accuracy']:.4f}")

    logger.info("\n📁 Saved Files:")
    logger.info("   ✅ models/best_model.pkl")
    logger.info("   ✅ models/artifacts/scaler.pkl")
    logger.info("   ✅ models/artifacts/label_encoders.pkl")
    logger.info("   ✅ models/model_metadata.json")
    logger.info("   ✅ docs/images/evaluation_report.md")
    logger.info("   ✅ docs/images/explainability_report.md")

    logger.info("\n🚀 Next Steps:")
    logger.info("   1. Review evaluation report: docs/images/evaluation_report.md")
    logger.info("   2. Start API: uvicorn api.main:app --reload")
    logger.info("   3. Test API: http://localhost:8000/docs")
    logger.info("   4. Create Power BI dashboard")
    logger.info("\n" + "=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\n⚠️  Training interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Error during training: {e}")
        import traceback

        traceback.print_exc()
