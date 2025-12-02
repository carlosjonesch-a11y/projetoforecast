"""
Modelos de Machine Learning.
"""

import numpy as np
from typing import Dict, Any


class XGBoostModel:
    """Wrapper para XGBoost."""
    
    def __init__(self, **kwargs):
        from xgboost import XGBRegressor
        self.model = XGBRegressor(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', 5),
            random_state=42,
            verbosity=0
        )
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    @property
    def feature_importances_(self):
        return self.model.feature_importances_


class LightGBMModel:
    """Wrapper para LightGBM."""
    
    def __init__(self, **kwargs):
        from lightgbm import LGBMRegressor
        self.model = LGBMRegressor(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', 5),
            random_state=42,
            verbosity=-1
        )
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    @property
    def feature_importances_(self):
        return self.model.feature_importances_


class RandomForestModel:
    """Wrapper para Random Forest."""
    
    def __init__(self, **kwargs):
        from sklearn.ensemble import RandomForestRegressor
        self.model = RandomForestRegressor(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', 10),
            random_state=42,
            n_jobs=-1
        )
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    @property
    def feature_importances_(self):
        return self.model.feature_importances_
