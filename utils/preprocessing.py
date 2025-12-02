"""
Pré-processamento de dados
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Optional


class FeatureEngineer:
    """Classe para criação de features."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_fitted = False
    
    def create_features(self, df: pd.DataFrame, target_col: str = 'Demanda') -> pd.DataFrame:
        """Cria features temporais."""
        df = df.copy()
        
        if 'Data' not in df.columns:
            raise ValueError("Coluna 'Data' não encontrada")
        
        df['Data'] = pd.to_datetime(df['Data'])
        
        # Features temporais
        df['dayofweek'] = df['Data'].dt.dayofweek
        df['month'] = df['Data'].dt.month
        df['day'] = df['Data'].dt.day
        df['quarter'] = df['Data'].dt.quarter
        df['dayofyear'] = df['Data'].dt.dayofyear
        df['weekofyear'] = df['Data'].dt.isocalendar().week.astype(int)
        
        # Lags
        for lag in [1, 7, 14, 30]:
            df[f'lag_{lag}'] = df[target_col].shift(lag)
        
        # Rolling
        for window in [7, 14, 30]:
            df[f'rolling_mean_{window}'] = df[target_col].shift(1).rolling(window=window).mean()
            df[f'rolling_std_{window}'] = df[target_col].shift(1).rolling(window=window).std()
        
        return df.dropna()
    
    def fit_transform(self, df: pd.DataFrame, target_col: str = 'Demanda') -> Tuple[pd.DataFrame, np.ndarray]:
        """Cria features e ajusta scaler."""
        df = self.create_features(df, target_col)
        
        feature_cols = [col for col in df.columns 
                       if col not in ['Data', target_col] 
                       and pd.api.types.is_numeric_dtype(df[col])]
        
        self.feature_columns = feature_cols
        
        X = df[feature_cols].astype(np.float64).values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = df[target_col].astype(np.float64).values
        
        X_scaled = self.scaler.fit_transform(X)
        self.is_fitted = True
        
        df_result = df.copy()
        df_result[feature_cols] = X_scaled
        
        return df_result, y
