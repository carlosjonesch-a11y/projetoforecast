"""
Dashboard de Previsão de Séries Temporais
Aplicativo principal com navegação multi-página
"""

import streamlit as st

st.set_page_config(
    page_title="Forecast Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">📊 Forecast Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Sistema de Previsão de Séries Temporais com Machine Learning</p>', unsafe_allow_html=True)

# Introdução
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🤖 11 Modelos de ML")
    st.markdown("""
    - XGBoost, LightGBM
    - Random Forest, Gradient Boosting
    - MLP Regressor, Ridge
    - ARIMA, SARIMA
    - Prophet, TBATS, Holt-Winters
    """)

with col2:
    st.markdown("### 📈 Análise Completa")
    st.markdown("""
    - Métricas detalhadas (MAE, RMSE, MAPE)
    - Gráficos interativos
    - Comparativo de modelos
    - Feature importance
    """)

with col3:
    st.markdown("### ⚙️ Flexibilidade")
    st.markdown("""
    - Upload de CSV/Excel
    - Granularidade: Diária/Horária/Turnos
    - Ensemble automático
    - Export de previsões
    """)

st.markdown("---")

# Status
st.markdown("### 📌 Status do Sistema")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if 'dados_carregados' in st.session_state and st.session_state.dados_carregados:
        st.success("✅ Dados carregados")
    else:
        st.warning("⏳ Aguardando dados")

with col2:
    if 'modelos_treinados' in st.session_state and st.session_state.modelos_treinados:
        st.success("✅ Modelos treinados")
    else:
        st.warning("⏳ Aguardando treinamento")

with col3:
    if 'previsoes_geradas' in st.session_state and st.session_state.previsoes_geradas:
        st.success("✅ Previsões geradas")
    else:
        st.warning("⏳ Aguardando previsões")

with col4:
    st.info("🔄 Sistema online")

# Instruções
st.markdown("---")
st.markdown("### 🚀 Como usar")

st.markdown("""
1. **📁 Upload Dados**: Faça upload do seu arquivo CSV ou Excel com dados de série temporal
2. **🤖 Treinamento**: Selecione e treine os modelos de previsão
3. **🔮 Previsões**: Gere previsões para o horizonte desejado
4. **📈 Comparativo**: Compare o desempenho dos modelos
5. **⚙️ Configurações**: Ajuste parâmetros do sistema
""")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888;'>"
    "Desenvolvido com ❤️ usando Streamlit | v1.0.0"
    "</div>",
    unsafe_allow_html=True
)
