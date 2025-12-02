"""
Classe base para modelos.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional


class BaseModel(ABC):
    """Classe base abstrata para todos os modelos."""
    
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.model = None
        self.is_fitted = False
        self.params = kwargs
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseModel':
        """Treina o modelo."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Faz previsões."""
        pass
    
    def get_params(self) -> Dict[str, Any]:
        """Retorna parâmetros do modelo."""
        return self.params
