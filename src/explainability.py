"""
Model Explainability Module
SHAP and LIME explanations
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

class ModelExplainer:
    """Model explainability with SHAP"""
    
    def __init__(self, model, X_train: np.ndarray, feature_names: list):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
    
    def create_explainer(self):
        """Create SHAP explainer"""
        try:
            # Tree-based explainer for tree models
            self.explainer = shap.TreeExplainer(self.model)
            logger.info("Created TreeExplainer")
        except:
            # Kernel explainer as fallback
            background = shap.sample(self.X_train, 100)
            self.explainer = shap.KernelExplainer(self.model.predict_proba, background)
            logger.info("Created KernelExplainer")
    
    def calculate_shap_values(self, X: np.ndarray):
        """Calculate SHAP values"""
        if self.explainer is None:
            self.create_explainer()
        
        self.shap_values = self.explainer.shap_values(X)
        
        # Handle multi-output
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[1]  # Get positive class
        
        logger.info("SHAP values calculated")
        return self.shap_values
    
    def plot_summary(self, X: np.ndarray, save_path: str = None):
        """Plot SHAP summary"""
        if self.shap_values is None:
            self.calculate_shap_values(X)
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(self.shap_values, X, feature_names=self.feature_names, show=False)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"SHAP summary saved to {save_path}")
        
        plt.close()
    
    def plot_feature_importance(self, X: np.ndarray, save_path: str = None):
        """Plot feature importance"""
        if self.shap_values is None:
            self.calculate_shap_values(X)
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(self.shap_values, X, feature_names=self.feature_names, 
                         plot_type="bar", show=False)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Feature importance saved to {save_path}")
        
        plt.close()
    
    def explain_prediction(self, X_instance: np.ndarray, index: int = 0) -> dict:
        """Explain individual prediction"""
        if self.shap_values is None:
            self.calculate_shap_values(X_instance)
        
        # Get SHAP values for this instance
        shap_vals = self.shap_values[index] if len(self.shap_values.shape) > 1 else self.shap_values
        
        # Get top features
        feature_impact = pd.DataFrame({
            'feature': self.feature_names,
            'shap_value': shap_vals,
            'feature_value': X_instance[index] if len(X_instance.shape) > 1 else X_instance
        })
        
        feature_impact['abs_shap'] = np.abs(feature_impact['shap_value'])
        feature_impact = feature_impact.sort_values('abs_shap', ascending=False)
        
        explanation = {
            'top_features': feature_impact.head(10).to_dict('records'),
            'prediction_confidence': float(np.abs(shap_vals.sum()))
        }
        
        return explanation
    
    def plot_waterfall(self, X_instance: np.ndarray, index: int = 0, save_path: str = None):
        """Plot waterfall chart for individual prediction"""
        if self.explainer is None:
            self.create_explainer()
        
        shap_values_instance = self.explainer(X_instance[index:index+1])
        
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(shap_values_instance[0], show=False)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Waterfall plot saved to {save_path}")
        
        plt.close()
    
    def get_feature_importance_df(self, X: np.ndarray) -> pd.DataFrame:
        """Get feature importance as DataFrame"""
        if self.shap_values is None:
            self.calculate_shap_values(X)
        
        importance = np.abs(self.shap_values).mean(axis=0)
        
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return df
    
    def generate_explanation_report(self, X_test: np.ndarray, output_dir: str = 'docs'):
        """Generate complete explanation report"""
        self.calculate_shap_values(X_test[:100])  # Sample for performance
        
        # Generate plots
        self.plot_summary(X_test[:100], f'{output_dir}/shap_summary.png')
        self.plot_feature_importance(X_test[:100], f'{output_dir}/feature_importance.png')
        
        # Get feature importance
        importance_df = self.get_feature_importance_df(X_test[:100])
        
        report = f"""
# Model Explainability Report

## Top 10 Most Important Features

| Rank | Feature | Importance |
|------|---------|------------|
"""
        for i, row in importance_df.head(10).iterrows():
            report += f"| {i+1} | {row['feature']} | {row['importance']:.4f} |\n"
        
        report += """
## SHAP Visualizations

### Feature Importance
![Feature Importance](feature_importance.png)

### SHAP Summary Plot
![SHAP Summary](shap_summary.png)

## Interpretation Guide

- **Positive SHAP values** (red): Push prediction towards churn
- **Negative SHAP values** (blue): Push prediction away from churn
- **Magnitude**: Indicates strength of feature impact

## Key Insights

Based on SHAP analysis, the top factors driving churn predictions are:

"""
        for i, row in importance_df.head(5).iterrows():
            report += f"{i+1}. **{row['feature']}**: High impact on predictions\n"
        
        with open(f'{output_dir}/explainability_report.md', 'w') as f:
            f.write(report)
        
        logger.info(f"Explainability report saved to {output_dir}/explainability_report.md")
        return report