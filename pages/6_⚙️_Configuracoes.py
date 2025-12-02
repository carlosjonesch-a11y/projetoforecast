"""
Página de Configurações
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

root_path = str(Path(__file__).parent.parent)
if root_path not in sys.path:
    sys.path.insert(0, root_path)

st.set_page_config(page_title="Configurações", page_icon="⚙️", layout="wide")

st.title("⚙️ Configurações")

# Status do sistema
st.markdown("### 📊 Status do Sistema")

col1, col2, col3 = st.columns(3)

with col1:
    if 'dados_carregados' in st.session_state and st.session_state.dados_carregados:
        st.success("✅ Dados carregados")
        if 'df' in st.session_state:
            st.write(f"Registros: {len(st.session_state.df)}")
    else:
        st.warning("⏳ Dados não carregados")

with col2:
    if 'modelos_treinados' in st.session_state and st.session_state.modelos_treinados:
        st.success("✅ Modelos treinados")
        if 'resultados_modelos' in st.session_state:
            st.write(f"Modelos: {len(st.session_state.resultados_modelos)}")
    else:
        st.warning("⏳ Modelos não treinados")

with col3:
    if 'previsoes_geradas' in st.session_state and st.session_state.previsoes_geradas:
        st.success("✅ Previsões geradas")
    else:
        st.warning("⏳ Previsões não geradas")

# Dados da sessão
st.markdown("---")
st.markdown("### 💾 Dados da Sessão")

if 'df' in st.session_state and st.session_state.df is not None:
    df = st.session_state.df
    
    with st.expander("📋 Visualizar dados carregados"):
        st.dataframe(df.head(20), width='stretch')
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Registros", len(df))
        with col2:
            st.metric("Colunas", len(df.columns))
        with col3:
            if 'Data' in df.columns:
                st.metric("Data início", df['Data'].min().strftime('%d/%m/%Y'))
        with col4:
            if 'Data' in df.columns:
                st.metric("Data fim", df['Data'].max().strftime('%d/%m/%Y'))

# Limpar dados
st.markdown("---")
st.markdown("### 🗑️ Limpar Dados")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🗑️ Limpar dados carregados"):
        if 'df' in st.session_state:
            st.session_state.df = None
            st.session_state.dados_carregados = False
            st.success("✅ Dados limpos")
            st.rerun()

with col2:
    if st.button("🗑️ Limpar modelos treinados"):
        if 'resultados_modelos' in st.session_state:
            st.session_state.resultados_modelos = {}
            st.session_state.modelos_treinados = False
            st.success("✅ Modelos limpos")
            st.rerun()

with col3:
    if st.button("🗑️ Limpar tudo"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.success("✅ Sessão limpa")
        st.rerun()

# Sobre
st.markdown("---")
st.markdown("### ℹ️ Sobre")

st.markdown("""
**Forecast Dashboard** v1.0.0

Sistema de previsão de séries temporais com Machine Learning.

**Modelos disponíveis:**
- XGBoost, LightGBM, Random Forest, Gradient Boosting
- MLP Regressor, Ridge Regression
- ARIMA, SARIMA, Prophet, TBATS, Holt-Winters

**Desenvolvido com:**
- Streamlit
- Scikit-learn
- XGBoost, LightGBM
- Prophet
- Statsmodels
- Plotly
""")
