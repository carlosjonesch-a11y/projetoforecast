"""
Página do Dashboard Principal
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Adicionar diretório raiz ao path
root_path = str(Path(__file__).parent.parent)
if root_path not in sys.path:
    sys.path.insert(0, root_path)

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")

st.title("📊 Dashboard")

# Verificar se há dados
if 'df' not in st.session_state or st.session_state.df is None:
    st.warning("⚠️ Nenhum dado carregado. Vá para **Upload Dados** primeiro.")
    st.stop()

df = st.session_state.df

# Métricas gerais
st.markdown("### 📈 Visão Geral dos Dados")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total de Registros", f"{len(df):,}")

with col2:
    if 'Data' in df.columns:
        dias = (df['Data'].max() - df['Data'].min()).days
        st.metric("Período (dias)", f"{dias:,}")

with col3:
    target_col = st.session_state.get('target_col', 'Demanda')
    if target_col in df.columns:
        st.metric("Média", f"{df[target_col].mean():,.2f}")

with col4:
    if target_col in df.columns:
        st.metric("Desvio Padrão", f"{df[target_col].std():,.2f}")

# Gráfico da série temporal
st.markdown("### 📉 Série Temporal")

if 'Data' in df.columns and target_col in df.columns:
    import plotly.express as px
    
    fig = px.line(df, x='Data', y=target_col, title=f"Série Temporal - {target_col}")
    fig.update_layout(
        xaxis_title="Data",
        yaxis_title=target_col,
        hovermode='x unified'
    )
    st.plotly_chart(fig, width='stretch')

# Estatísticas
st.markdown("### 📊 Estatísticas Descritivas")

if target_col in df.columns:
    stats = df[target_col].describe()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(stats.to_frame().T, width='stretch')
    
    with col2:
        # Histograma
        fig_hist = px.histogram(df, x=target_col, nbins=30, title=f"Distribuição - {target_col}")
        st.plotly_chart(fig_hist, width='stretch')

# Informações do modelo se treinado
if 'modelos_treinados' in st.session_state and st.session_state.modelos_treinados:
    st.markdown("### 🤖 Modelos Treinados")
    
    if 'resultados_modelos' in st.session_state:
        resultados = st.session_state.resultados_modelos
        
        metrics_data = []
        for nome, res in resultados.items():
            if 'metrics' in res:
                metrics_data.append({
                    'Modelo': nome,
                    'MAE': res['metrics'].get('mae', 0),
                    'RMSE': res['metrics'].get('rmse', 0),
                    'MAPE (%)': res['metrics'].get('mape', 0) * 100
                })
        
        if metrics_data:
            df_metrics = pd.DataFrame(metrics_data)
            df_metrics = df_metrics.sort_values('MAE')
            st.dataframe(df_metrics, width='stretch', hide_index=True)
