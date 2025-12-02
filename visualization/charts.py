"""
Gráficos interativos com Plotly.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List, Optional


def plot_time_series(df: pd.DataFrame, 
                     date_col: str = 'Data', 
                     value_col: str = 'Demanda',
                     title: str = 'Série Temporal') -> go.Figure:
    """
    Cria gráfico de série temporal.
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df[date_col],
        y=df[value_col],
        mode='lines',
        name=value_col,
        line=dict(color='#2C3E50', width=2)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Data',
        yaxis_title=value_col,
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig


def plot_forecast(df_historical: pd.DataFrame,
                  df_forecast: pd.DataFrame,
                  date_col: str = 'Data',
                  value_col: str = 'Demanda',
                  forecast_col: str = 'Previsao',
                  title: str = 'Previsão') -> go.Figure:
    """
    Cria gráfico com histórico e previsão.
    """
    fig = go.Figure()
    
    # Histórico
    fig.add_trace(go.Scatter(
        x=df_historical[date_col],
        y=df_historical[value_col],
        mode='lines',
        name='Histórico',
        line=dict(color='#2C3E50', width=2)
    ))
    
    # Previsão
    fig.add_trace(go.Scatter(
        x=df_forecast[date_col],
        y=df_forecast[forecast_col],
        mode='lines',
        name='Previsão',
        line=dict(color='#E74C3C', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Data',
        yaxis_title=value_col,
        hovermode='x unified',
        template='plotly_white',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    return fig


def plot_model_comparison(metrics_df: pd.DataFrame,
                          metric: str = 'MAE',
                          title: str = 'Comparação de Modelos') -> go.Figure:
    """
    Cria gráfico de barras comparando modelos.
    """
    fig = px.bar(
        metrics_df,
        x='Modelo',
        y=metric,
        title=title,
        color=metric,
        color_continuous_scale='RdYlGn_r'
    )
    
    return fig
