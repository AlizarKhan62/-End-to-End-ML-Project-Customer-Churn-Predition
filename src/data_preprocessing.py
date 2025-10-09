"""
Data Preprocessing Module
Handles data cleaning, validation, and initial transformations
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from sklearn.preprocessing import LabelEncoder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Comprehensive data preprocessing"""
    
    def __init__(self, config: dict):
        self.config = config
        self.label_encoders = {}
        self.feature_names = None
        self.id_columns = ['customerID']  # ID columns to remove
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load and validate data"""
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        return df
    
    def remove_id_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove ID and non-predictive columns"""
        df = df.copy()
        cols_to_remove = [col for col in self.id_columns if col in df.columns]
        
        if cols_to_remove:
            df = df.drop(columns=cols_to_remove)
            logger.info(f"Removed ID columns: {cols_to_remove}")
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values"""
        method = self.config.get('handle_missing', 'drop')
        
        missing_count = df.isnull().sum().sum()
        if missing_count == 0:
            logger.info("No missing values found")
            return df
        
        logger.info(f"Found {missing_count} missing values")
        
        if method == 'drop':
            df = df.dropna()
            logger.info("Dropped rows with missing values")
        elif method == 'mean':
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
            logger.info("Filled missing numerical values with mean")
        elif method == 'median':
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
            logger.info("Filled missing numerical values with median")
        elif method == 'mode':
            for col in df.columns:
                df[col].fillna(df[col].mode()[0], inplace=True)
            logger.info("Filled missing values with mode")
        
        return df
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows"""
        initial_count = len(df)
        df = df.drop_duplicates()
        removed = initial_count - len(df)
        if removed > 0:
            logger.info(f"Removed {removed} duplicate rows")
        return df
    
    def encode_target(self, df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, np.ndarray]:
        """Encode target variable"""
        if df[target_col].dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(df[target_col])
            self.label_encoders['target'] = le
            logger.info(f"Target encoded: {dict(enumerate(le.classes_))}")
        else:
            y = df[target_col].values
        
        X = df.drop(target_col, axis=1)
        return X, y
    
    def encode_categorical(self, df: pd.DataFrame, categorical_cols: List[str]) -> pd.DataFrame:
        """Encode categorical features"""
        df = df.copy()
        
        for col in categorical_cols:
            if col not in df.columns:
                logger.warning(f"Column {col} not found, skipping")
                continue
            
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
            logger.info(f"Encoded {col}: {len(le.classes_)} categories")
        
        return df
    
    def remove_outliers(self, df: pd.DataFrame, numerical_cols: List[str], 
                       method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """Remove outliers using IQR or Z-score method"""
        df = df.copy()
        initial_count = len(df)
        
        if method == 'iqr':
            for col in numerical_cols:
                if col not in df.columns:
                    continue
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR
                df = df[(df[col] >= lower) & (df[col] <= upper)]
        
        elif method == 'zscore':
            from scipy import stats
            for col in numerical_cols:
                if col not in df.columns:
                    continue
                z_scores = np.abs(stats.zscore(df[col]))
                df = df[z_scores < threshold]
        
        removed = initial_count - len(df)
        if removed > 0:
            logger.info(f"Removed {removed} outliers ({removed/initial_count*100:.2f}%)")
        
        return df
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate data quality"""
        issues = []
        
        # Check for negative values in specific columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if (df[col] < 0).any():
                issues.append(f"Negative values found in {col}")
        
        # Check for extremely high values
        for col in numerical_cols:
            if df[col].max() > df[col].mean() + 10 * df[col].std():
                issues.append(f"Extreme outliers in {col}")
        
        # Check for constant columns
        for col in df.columns:
            if df[col].nunique() == 1:
                issues.append(f"Constant column: {col}")
        
        if issues:
            logger.warning(f"Data validation issues: {issues}")
            return False
        
        logger.info("Data validation passed")
        return True
    
    def preprocess(self, df: pd.DataFrame, target_col: str = 'Churn', 
                   fit: bool = True) -> Tuple[pd.DataFrame, np.ndarray]:
        """Complete preprocessing pipeline"""
        logger.info("Starting preprocessing pipeline")
        
        # Remove ID columns FIRST
        df = self.remove_id_columns(df)
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Remove duplicates
        df = self.remove_duplicates(df)
        
        # Encode target
        X, y = self.encode_target(df, target_col)
        
        # Get categorical columns
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        # Encode categorical features
        if fit:
            X = self.encode_categorical(X, categorical_cols)
            self.feature_names = X.columns.tolist()
        else:
            # Use existing encoders
            for col in categorical_cols:
                if col in self.label_encoders:
                    X[col] = self.label_encoders[col].transform(X[col].astype(str))
        
        # Validate
        self.validate_data(X)
        
        logger.info(f"Preprocessing complete: {X.shape}")
        return X, y
    
    def save_encoders(self, filepath: str):
        """Save label encoders"""
        import joblib
        joblib.dump(self.label_encoders, filepath)
        logger.info(f"Saved encoders to {filepath}")
    
    def load_encoders(self, filepath: str):
        """Load label encoders"""
        import joblib
        self.label_encoders = joblib.load(filepath)
        logger.info(f"Loaded encoders from {filepath}")