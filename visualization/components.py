"""
Componentes reutilizáveis para o dashboard.
"""

import streamlit as st
from typing import Any, Optional


def metric_card(title: str, 
                value: Any, 
                delta: float = None,
                delta_color: str = "normal",
                help_text: str = None) -> None:
    """
    Exibe um card de métrica.
    """
    st.metric(
        label=title,
        value=value,
        delta=f"{delta:+.2f}%" if delta is not None else None,
        delta_color=delta_color,
        help=help_text
    )


def data_quality_indicator(score: float) -> None:
    """
    Exibe indicador de qualidade dos dados.
    """
    if score >= 90:
        color = "🟢"
        status = "Excelente"
    elif score >= 70:
        color = "🟡"
        status = "Bom"
    elif score >= 50:
        color = "🟠"
        status = "Regular"
    else:
        color = "🔴"
        status = "Ruim"
    
    st.markdown(f"{color} **Qualidade dos dados:** {status} ({score:.1f}%)")


def progress_bar(current: int, total: int, text: str = "") -> None:
    """
    Exibe barra de progresso.
    """
    progress = current / total if total > 0 else 0
    st.progress(progress, text=text)


def model_status_badge(model_name: str, status: str) -> None:
    """
    Exibe badge de status do modelo.
    """
    colors = {
        'trained': '🟢',
        'training': '🟡',
        'error': '🔴',
        'pending': '⚪'
    }
    
    color = colors.get(status, '⚪')
    st.markdown(f"{color} **{model_name}**")
