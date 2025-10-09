"""
Feature Engineering Module
Creates advanced features for better model performance
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Advanced feature engineering"""
    
    def __init__(self, config: dict):
        self.config = config
        self.poly_features = None
        self.feature_names = []
        
    def create_interaction_features(self, df: pd.DataFrame, 
                                    numerical_cols: list) -> pd.DataFrame:
        """Create interaction features between numerical columns"""
        df = df.copy()
        
        for i, col1 in enumerate(numerical_cols):
            for col2 in numerical_cols[i+1:]:
                if col1 in df.columns and col2 in df.columns:
                    new_col = f"{col1}_x_{col2}"
                    df[new_col] = df[col1] * df[col2]
                    logger.info(f"Created interaction: {new_col}")
        
        return df
    
    def create_polynomial_features(self, df: pd.DataFrame, 
                                   numerical_cols: list, 
                                   degree: int = 2) -> pd.DataFrame:
        """Create polynomial features"""
        df = df.copy()
        
        if self.poly_features is None:
            self.poly_features = PolynomialFeatures(
                degree=degree, 
                include_bias=False,
                interaction_only=False
            )
            
            poly_array = self.poly_features.fit_transform(df[numerical_cols])
            poly_feature_names = self.poly_features.get_feature_names_out(numerical_cols)
        else:
            poly_array = self.poly_features.transform(df[numerical_cols])
            poly_feature_names = self.poly_features.get_feature_names_out(numerical_cols)
        
        # Add polynomial features
        poly_df = pd.DataFrame(
            poly_array[:, len(numerical_cols):],  # Skip original features
            columns=poly_feature_names[len(numerical_cols):],
            index=df.index
        )
        
        df = pd.concat([df, poly_df], axis=1)
        logger.info(f"Created {len(poly_df.columns)} polynomial features")
        
        return df
    
    def create_tenure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create tenure-based features"""
        df = df.copy()
        
        if 'tenure' in df.columns:
            df['tenure_squared'] = df['tenure'] ** 2
            df['tenure_sqrt'] = np.sqrt(df['tenure'])
            df['tenure_log'] = np.log1p(df['tenure'])
            
            # Tenure bins
            df['tenure_group'] = pd.cut(
                df['tenure'], 
                bins=[0, 12, 24, 48, 72, 100], 
                labels=['0-1yr', '1-2yr', '2-4yr', '4-6yr', '6+yr']
            ).astype(str)
            
            logger.info("Created tenure-based features")
        
        return df
    
    def create_charge_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create charge-based features"""
        df = df.copy()
        
        if 'monthly_charges' in df.columns and 'total_charges' in df.columns:
            # Ratio features
            df['avg_monthly_charges'] = df['total_charges'] / (df['tenure'] + 1)
            df['charge_to_tenure_ratio'] = df['monthly_charges'] / (df['tenure'] + 1)
            
            # Difference from average
            df['charge_diff_from_avg'] = df['monthly_charges'] - df['monthly_charges'].mean()
            
            # Price bins
            df['price_category'] = pd.cut(
                df['monthly_charges'],
                bins=[0, 30, 60, 90, 150],
                labels=['Low', 'Medium', 'High', 'Premium']
            ).astype(str)
            
            logger.info("Created charge-based features")
        
        return df
    
    def create_service_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create service-related features"""
        df = df.copy()
        
        service_cols = [
            'online_security', 'online_backup', 'device_protection',
            'tech_support', 'streaming_tv', 'streaming_movies'
        ]
        
        # Count active services
        active_services = []
        for col in service_cols:
            if col in df.columns:
                # Assuming 'Yes'/'No' or 1/0 encoding
                if df[col].dtype == 'object':
                    active_services.append((df[col] == 'Yes').astype(int))
                else:
                    active_services.append(df[col])
        
        if active_services:
            df['total_services'] = sum(active_services)
            df['service_usage_score'] = df['total_services'] / len(active_services)
            logger.info("Created service-based features")
        
        return df
    
    def create_contract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create contract-related features"""
        df = df.copy()
        
        if 'contract_type' in df.columns:
            # Contract risk (month-to-month = higher risk)
            contract_risk_map = {
                'Month-to-month': 2,
                'One year': 1,
                'Two year': 0
            }
            df['contract_risk'] = df['contract_type'].map(
                lambda x: contract_risk_map.get(x, 1)
            )
            
            logger.info("Created contract-based features")
        
        return df
    
    def create_clv_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create Customer Lifetime Value features"""
        df = df.copy()
        
        if all(col in df.columns for col in ['tenure', 'monthly_charges']):
            # Estimated CLV
            df['estimated_clv'] = df['tenure'] * df['monthly_charges']
            
            # Potential CLV (assuming average customer lifetime)
            avg_lifetime = 36  # months
            df['potential_clv'] = avg_lifetime * df['monthly_charges']
            
            # CLV at risk
            df['clv_at_risk'] = df['estimated_clv'] * 0.5  # Assuming 50% retention
            
            logger.info("Created CLV features")
        
        return df
    
    def create_statistical_features(self, df: pd.DataFrame, 
                                   numerical_cols: list) -> pd.DataFrame:
        """Create statistical aggregation features"""
        df = df.copy()
        
        # Z-scores
        for col in numerical_cols:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                df[f'{col}_zscore'] = (df[col] - mean) / (std + 1e-10)
        
        logger.info("Created statistical features")
        return df
    
    def engineer_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Complete feature engineering pipeline"""
        logger.info("Starting feature engineering")
        
        # Get numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Create tenure features
        df = self.create_tenure_features(df)
        
        # Create charge features
        df = self.create_charge_features(df)
        
        # Create service features
        df = self.create_service_features(df)
        
        # Create contract features
        df = self.create_contract_features(df)
        
        # Create CLV features
        df = self.create_clv_features(df)
        
        # Create interactions if enabled
        if self.config.get('create_interactions', False):
            base_numerical = ['tenure', 'monthly_charges', 'total_charges']
            available_numerical = [col for col in base_numerical if col in df.columns]
            df = self.create_interaction_features(df, available_numerical)
        
        # Create polynomial features if enabled
        if self.config.get('polynomial_degree', 0) > 1:
            base_numerical = ['tenure', 'monthly_charges']
            available_numerical = [col for col in base_numerical if col in df.columns]
            if fit:
                df = self.create_polynomial_features(
                    df, available_numerical, 
                    degree=self.config['polynomial_degree']
                )
        
        # Create statistical features
        new_numerical = df.select_dtypes(include=[np.number]).columns.tolist()
        df = self.create_statistical_features(df, new_numerical[:10])  # Limit to avoid too many
        
        self.feature_names = df.columns.tolist()
        logger.info(f"Feature engineering complete: {len(self.feature_names)} features")
        
        return df
    
    def select_features(self, df: pd.DataFrame, y: np.ndarray, 
                       method: str = 'mutual_info', k: int = 30) -> pd.DataFrame:
        """Select top k features"""
        from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif
        
        if method == 'mutual_info':
            selector = SelectKBest(mutual_info_classif, k=k)
        elif method == 'f_classif':
            selector = SelectKBest(f_classif, k=k)
        else:
            logger.warning(f"Unknown method {method}, skipping feature selection")
            return df
        
        X_selected = selector.fit_transform(df, y)
        selected_features = df.columns[selector.get_support()].tolist()
        
        logger.info(f"Selected {len(selected_features)} features using {method}")
        return df[selected_features]