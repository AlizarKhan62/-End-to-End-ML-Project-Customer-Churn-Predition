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

    def create_interaction_features(self, df: pd.DataFrame, numerical_cols: list) -> pd.DataFrame:
        """Create interaction features between numerical columns"""
        df = df.copy()
        for i, col1 in enumerate(numerical_cols):
            for col2 in numerical_cols[i + 1:]:
                if col1 in df.columns and col2 in df.columns:
                    new_col = f"{col1}_x_{col2}"
                    df[new_col] = df[col1] * df[col2]
                    logger.info(f"Created interaction: {new_col}")
        return df

    def create_polynomial_features(self, df: pd.DataFrame, numerical_cols: list, degree: int = 2) -> pd.DataFrame:
        """Create polynomial features"""
        df = df.copy()
        if self.poly_features is None:
            self.poly_features = PolynomialFeatures(degree=degree, include_bias=False)
            poly_array = self.poly_features.fit_transform(df[numerical_cols])
        else:
            poly_array = self.poly_features.transform(df[numerical_cols])

        poly_feature_names = self.poly_features.get_feature_names_out(numerical_cols)
        poly_df = pd.DataFrame(
            poly_array[:, len(numerical_cols):],  # Skip originals
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

            # Tenure bins converted to numeric
            df['tenure_group'] = pd.cut(
                df['tenure'],
                bins=[0, 12, 24, 48, 72, 100],
                labels=['0-1yr', '1-2yr', '2-4yr', '4-6yr', '6+yr']
            ).astype(str)

            tenure_map = {
                '0-1yr': 1,
                '1-2yr': 2,
                '2-4yr': 3,
                '4-6yr': 4,
                '6+yr': 5
            }
            df['tenure_group'] = df['tenure_group'].map(tenure_map)

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
        
        # Price bins (numeric encoding instead of strings)
        price_bins = [0, 30, 60, 90, 150]
        price_labels = ['Low', 'Medium', 'High', 'Premium']
        df['price_category'] = pd.cut(df['monthly_charges'], bins=price_bins, labels=price_labels)
        
        # Map to numeric values
        price_map = {'Low': 1, 'Medium': 2, 'High': 3, 'Premium': 4}
        df['price_category'] = df['price_category'].astype(str).map(price_map)
        
        logger.info("Created charge-based features (numeric price category)")
    
    return df



    def create_service_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create service-related features"""
        df = df.copy()
        service_cols = [
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]

        active_services = []
        for col in service_cols:
            if col in df.columns:
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
        contract_col = 'Contract' if 'Contract' in df.columns else 'contract_type'

        if contract_col in df.columns:
            contract_risk_map = {
                'Month-to-month': 2,
                'One year': 1,
                'Two year': 0
            }
            df['contract_risk'] = df[contract_col].map(lambda x: contract_risk_map.get(x, 1))
            logger.info("Created contract-based features")
        return df

    def create_clv_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create Customer Lifetime Value features"""
        df = df.copy()
        if all(col in df.columns for col in ['tenure', 'MonthlyCharges']):
            df['estimated_clv'] = df['tenure'] * df['MonthlyCharges']
            avg_lifetime = 36
            df['potential_clv'] = avg_lifetime * df['MonthlyCharges']
            df['clv_at_risk'] = df['estimated_clv'] * 0.5
            logger.info("Created CLV features")
        return df

    def create_statistical_features(self, df: pd.DataFrame, numerical_cols: list) -> pd.DataFrame:
        """Create statistical aggregation features"""
        df = df.copy()
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

        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        df = self.create_tenure_features(df)
        df = self.create_charge_features(df)
        df = self.create_service_features(df)
        df = self.create_contract_features(df)
        df = self.create_clv_features(df)

        if self.config.get('create_interactions', False):
            base_numerical = ['tenure', 'MonthlyCharges', 'TotalCharges']
            available = [col for col in base_numerical if col in df.columns]
            df = self.create_interaction_features(df, available)

        if self.config.get('polynomial_degree', 0) > 1:
            base_numerical = ['tenure', 'MonthlyCharges']
            available = [col for col in base_numerical if col in df.columns]
            if fit:
                df = self.create_polynomial_features(df, available, degree=self.config['polynomial_degree'])

        new_numerical = df.select_dtypes(include=[np.number]).columns.tolist()
        df = self.create_statistical_features(df, new_numerical[:10])

        self.feature_names = df.columns.tolist()
        logger.info(f"Feature engineering complete: {len(self.feature_names)} features")
        return df
