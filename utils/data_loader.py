"""
Data Loader - Carregamento e validação de dados
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
from pathlib import Path


class DataLoader:
    """Classe para carregar e validar dados."""
    
    def __init__(self):
        self.supported_formats = ['.csv', '.xlsx', '.xls']
    
    def load_csv(self, filepath: str, **kwargs) -> pd.DataFrame:
        """Carrega arquivo CSV."""
        try:
            df = pd.read_csv(filepath, sep=';', encoding='utf-8', **kwargs)
            if len(df.columns) == 1:
                df = pd.read_csv(filepath, sep=',', encoding='utf-8', **kwargs)
        except:
            df = pd.read_csv(filepath, encoding='latin-1', **kwargs)
        return df
    
    def load_excel(self, filepath: str, **kwargs) -> pd.DataFrame:
        """Carrega arquivo Excel."""
        return pd.read_excel(filepath, **kwargs)
    
    def load_file(self, filepath: str, **kwargs) -> pd.DataFrame:
        """Carrega arquivo baseado na extensão."""
        path = Path(filepath)
        
        if path.suffix.lower() == '.csv':
            return self.load_csv(filepath, **kwargs)
        elif path.suffix.lower() in ['.xlsx', '.xls']:
            return self.load_excel(filepath, **kwargs)
        else:
            raise ValueError(f"Formato não suportado: {path.suffix}")


def validate_dataframe(df: pd.DataFrame, 
                       required_cols: Optional[List[str]] = None) -> Tuple[bool, str]:
    """
    Valida um DataFrame.
    
    Returns:
        Tuple[bool, str]: (é_válido, mensagem)
    """
    if df is None or df.empty:
        return False, "DataFrame está vazio"
    
    if required_cols:
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            return False, f"Colunas faltando: {missing}"
    
    return True, "DataFrame válido"
