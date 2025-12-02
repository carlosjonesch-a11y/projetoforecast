"""
Modelo Ensemble.
"""

import numpy as np
from typing import Dict, List


class EnsembleModel:
    """Combina previsões de múltiplos modelos."""
    
    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {}
        self.predictions = {}
    
    def add_prediction(self, model_name: str, prediction: np.ndarray):
        """Adiciona previsão de um modelo."""
        self.predictions[model_name] = prediction
    
    def predict(self, method: str = 'mean') -> np.ndarray:
        """
        Combina previsões.
        
        Args:
            method: 'mean', 'median', ou 'weighted'
        """
        if not self.predictions:
            raise ValueError("Nenhuma previsão adicionada")
        
        preds = list(self.predictions.values())
        
        if method == 'mean':
            return np.mean(preds, axis=0)
        elif method == 'median':
            return np.median(preds, axis=0)
        elif method == 'weighted':
            if not self.weights:
                return np.mean(preds, axis=0)
            
            weighted_sum = np.zeros_like(preds[0])
            total_weight = 0
            
            for name, pred in self.predictions.items():
                weight = self.weights.get(name, 1.0)
                weighted_sum += pred * weight
                total_weight += weight
            
            return weighted_sum / total_weight
        else:
            raise ValueError(f"Método desconhecido: {method}")
