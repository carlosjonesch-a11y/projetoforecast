"""
Helpers - Funções auxiliares
"""

import pandas as pd
import numpy as np
from typing import Any, Optional


def format_number(value: float, decimals: int = 2) -> str:
    """Formata número com separador de milhares."""
    return f"{value:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def format_percentage(value: float, decimals: int = 2) -> str:
    """Formata porcentagem."""
    return f"{value * 100:.{decimals}f}%"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divisão segura que evita divisão por zero."""
    if denominator == 0:
        return default
    return numerator / denominator


def calculate_data_quality_score(df: pd.DataFrame) -> float:
    """
    Calcula score de qualidade dos dados.
    
    Returns:
        float: Score entre 0 e 100
    """
    if df is None or df.empty:
        return 0.0
    
    scores = []
    
    # 1. Completude (% de valores não nulos)
    completeness = (1 - df.isnull().sum().sum() / df.size) * 100
    scores.append(completeness)
    
    # 2. Unicidade de datas (se houver coluna Data)
    if 'Data' in df.columns:
        uniqueness = (df['Data'].nunique() / len(df)) * 100
        scores.append(uniqueness)
    
    # 3. Valores válidos (não infinitos)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        inf_count = np.isinf(df[numeric_cols]).sum().sum()
        validity = (1 - inf_count / df[numeric_cols].size) * 100
        scores.append(validity)
    
    return np.mean(scores) if scores else 0.0
