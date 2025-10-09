"""
Model Evaluation Module
Comprehensive model evaluation and metrics
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self, config: dict):
        self.config = config
        self.threshold = config['evaluation'].get('threshold', 0.5)
    
    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate model and return metrics"""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'accuracy': accuracy_score(y_test, y_pred)
        }
        
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics
    
    def get_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Get confusion matrix"""
        return confusion_matrix(y_true, y_pred)
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            save_path: str = None):
        """Plot confusion matrix"""
        cm = self.get_confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Not Churn', 'Churn'],
                   yticklabels=['Not Churn', 'Churn'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.close()
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                      save_path: str = None):
        """Plot ROC curve"""
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"ROC curve saved to {save_path}")
        
        plt.close()
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                   save_path: str = None):
        """Plot Precision-Recall curve"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"PR curve saved to {save_path}")
        
        plt.close()
    
    def get_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """Get classification report"""
        return classification_report(y_true, y_pred, 
                                    target_names=['Not Churn', 'Churn'])
    
    def calculate_business_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                  avg_customer_value: float = 1200) -> dict:
        """Calculate business impact metrics"""
        cm = self.get_confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Business metrics
        customers_saved = tp  # True positives
        false_alarms = fp  # False positives
        missed_churners = fn  # False negatives
        
        revenue_saved = customers_saved * avg_customer_value
        potential_revenue_loss = missed_churners * avg_customer_value
        campaign_cost = (tp + fp) * 50  # Assuming $50 per retention campaign
        
        roi = ((revenue_saved - campaign_cost) / campaign_cost) * 100 if campaign_cost > 0 else 0
        
        business_metrics = {
            'customers_saved': int(customers_saved),
            'false_alarms': int(false_alarms),
            'missed_churners': int(missed_churners),
            'revenue_saved': float(revenue_saved),
            'potential_revenue_loss': float(potential_revenue_loss),
            'campaign_cost': float(campaign_cost),
            'roi_percentage': float(roi)
        }
        
        logger.info(f"Business metrics: Revenue saved ${revenue_saved:,.2f}, ROI {roi:.2f}%")
        return business_metrics
    
    def find_optimal_threshold(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                             metric: str = 'f1') -> float:
        """Find optimal classification threshold"""
        thresholds = np.linspace(0, 1, 100)
        scores = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_true, y_pred)
            elif metric == 'precision':
                score = precision_score(y_true, y_pred)
            elif metric == 'recall':
                score = recall_score(y_true, y_pred)
            else:
                score = f1_score(y_true, y_pred)
            
            scores.append(score)
        
        optimal_idx = np.argmax(scores)
        optimal_threshold = thresholds[optimal_idx]
        
        logger.info(f"Optimal threshold for {metric}: {optimal_threshold:.3f}")
        return optimal_threshold
    
    def generate_evaluation_report(self, model, X_test: np.ndarray, 
                                  y_test: np.ndarray, output_dir: str = 'docs'):
        """Generate complete evaluation report"""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Get all metrics
        metrics = self.evaluate_model(model, X_test, y_test)
        business_metrics = self.calculate_business_metrics(y_test, y_pred)
        classification_rep = self.get_classification_report(y_test, y_pred)
        
        # Plot visualizations
        self.plot_confusion_matrix(y_test, y_pred, f'{output_dir}/confusion_matrix.png')
        self.plot_roc_curve(y_test, y_pred_proba, f'{output_dir}/roc_curve.png')
        self.plot_precision_recall_curve(y_test, y_pred_proba, f'{output_dir}/pr_curve.png')
        
        # Create report
        report = f"""
# Model Evaluation Report

## Performance Metrics
- **ROC-AUC Score**: {metrics['roc_auc']:.4f}
- **Precision**: {metrics['precision']:.4f}
- **Recall**: {metrics['recall']:.4f}
- **F1-Score**: {metrics['f1']:.4f}
- **Accuracy**: {metrics['accuracy']:.4f}

## Classification Report
```
{classification_rep}
```

## Business Impact
- **Customers Saved**: {business_metrics['customers_saved']}
- **Revenue Saved**: ${business_metrics['revenue_saved']:,.2f}
- **Campaign Cost**: ${business_metrics['campaign_cost']:,.2f}
- **ROI**: {business_metrics['roi_percentage']:.2f}%
- **Missed Churners**: {business_metrics['missed_churners']}
- **Potential Revenue Loss**: ${business_metrics['potential_revenue_loss']:,.2f}

## Visualizations
![Confusion Matrix](confusion_matrix.png)
![ROC Curve](roc_curve.png)
![Precision-Recall Curve](pr_curve.png)
"""
        
        with open(f'{output_dir}/evaluation_report.md', 'w') as f:
            f.write(report)
        
        logger.info(f"Evaluation report saved to {output_dir}/evaluation_report.md")
        return report