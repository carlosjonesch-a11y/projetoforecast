"""
Página de Comparativo de Modelos
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

root_path = str(Path(__file__).parent.parent)
if root_path not in sys.path:
    sys.path.insert(0, root_path)

st.set_page_config(page_title="Comparativo", page_icon="📈", layout="wide")

st.title("📈 Comparativo de Modelos")

# Verificar se há modelos
if 'modelos_treinados' not in st.session_state or not st.session_state.modelos_treinados:
    st.warning("⚠️ Nenhum modelo treinado. Vá para **Treinamento** primeiro.")
    st.stop()

# Detectar modo de previsão
modo_previsao = st.session_state.get('modo_previsao', 'agregado')

# ========== MODO POR GRUPOS (Local/Sublocal/Hora) ==========
if modo_previsao in ['local', 'grupo'] and 'resultados_por_grupo' in st.session_state:
    resultados_por_grupo = st.session_state.resultados_por_grupo
    metricas_por_grupo = st.session_state.metricas_por_grupo
    grupos = list(resultados_por_grupo.keys())
    modo_agrupamento = st.session_state.get('modo_agrupamento', 'Por Local')
    
    st.info(f"🏢 Modo: **{modo_agrupamento}** | {len(grupos)} grupos")
    
    # Selecionar grupo para comparar
    grupo_sel = st.selectbox("Selecione o grupo para comparar:", grupos)
    
    resultados = resultados_por_grupo[grupo_sel]
    df_metrics = metricas_por_grupo[grupo_sel]

# ========== MODO POR TURNO ==========
elif modo_previsao == 'turno' and 'resultados_por_turno' in st.session_state:
    resultados_por_turno = st.session_state.resultados_por_turno
    metricas_por_turno = st.session_state.metricas_por_turno
    turnos = list(resultados_por_turno.keys())
    
    st.info(f"🕐 Modo: **Por Turno** | Turnos: {', '.join(turnos)}")
    
    # Selecionar turno para comparar
    turno_sel = st.selectbox("Selecione o turno para comparar:", turnos)
    
    resultados = resultados_por_turno[turno_sel]
    df_metrics = metricas_por_turno[turno_sel]

# ========== MODO AGREGADO ==========
elif 'resultados_modelos' in st.session_state:
    resultados = st.session_state.resultados_modelos
    
    if not resultados:
        st.warning("⚠️ Nenhum resultado disponível.")
        st.stop()
    
    st.info("📊 Modo: **Agregado**")
    
    # Calcular métricas se não existir
    if 'df_metrics' in st.session_state:
        df_metrics = st.session_state.df_metrics
    else:
        metrics_data = []
        for nome, res in resultados.items():
            metrics_data.append({
                'Modelo': nome,
                'MAE': res['metrics']['mae'],
                'RMSE': res['metrics']['rmse'],
                'MAPE (%)': res['metrics']['mape'] * 100,
                'R²': res['metrics']['r2']
            })
        df_metrics = pd.DataFrame(metrics_data).sort_values('MAE')

else:
    st.warning("⚠️ Nenhum resultado de modelo encontrado.")
    st.stop()

st.markdown(f"**Modelos disponíveis:** {len(resultados)}")

# Métricas
st.markdown("---")
st.markdown("### 📊 Métricas de Desempenho")

# Garantir que df_metrics está ordenado
df_metrics = df_metrics.sort_values('MAE').reset_index(drop=True)

# Destacar melhor modelo
best_model = df_metrics.iloc[0]['Modelo']
st.success(f"🏆 Melhor modelo: **{best_model}** (menor MAE)")

# Tabela formatada
highlight_cols = [col for col in ['MAE', 'RMSE', 'MAPE (%)'] if col in df_metrics.columns]
st.dataframe(
    df_metrics.style.highlight_min(subset=highlight_cols, color='lightgreen'),
    width='stretch',
    hide_index=True
)

# Gráficos
st.markdown("---")
st.markdown("### 📉 Visualizações")

import plotly.express as px
import plotly.graph_objects as go

col1, col2 = st.columns(2)

with col1:
    # Gráfico de barras - MAE
    fig_mae = px.bar(
        df_metrics,
        x='Modelo',
        y='MAE',
        title='MAE por Modelo',
        color='MAE',
        color_continuous_scale='RdYlGn_r'
    )
    st.plotly_chart(fig_mae, width='stretch')

with col2:
    # Gráfico de barras - RMSE
    fig_rmse = px.bar(
        df_metrics,
        x='Modelo',
        y='RMSE',
        title='RMSE por Modelo',
        color='RMSE',
        color_continuous_scale='RdYlGn_r'
    )
    st.plotly_chart(fig_rmse, width='stretch')

# Radar chart
st.markdown("### 🎯 Comparação Radar")

# Normalizar métricas para radar (inverter para que menor seja melhor)
df_radar = df_metrics.copy()

# Verificar quais colunas existem
cols_metricas = [col for col in ['MAE', 'RMSE', 'MAPE (%)'] if col in df_radar.columns]

for col in cols_metricas:
    max_val = df_radar[col].max()
    if max_val > 0:
        df_radar[f'{col}_norm'] = 1 - (df_radar[col] / max_val)
    else:
        df_radar[f'{col}_norm'] = 0

fig_radar = go.Figure()

for i, row in df_radar.iterrows():
    r_values = [row.get(f'{col}_norm', 0) for col in cols_metricas]
    theta_values = [col.replace(' (%)', '') for col in cols_metricas]
    
    fig_radar.add_trace(go.Scatterpolar(
        r=r_values,
        theta=theta_values,
        fill='toself',
        name=row['Modelo']
    ))

fig_radar.update_layout(
    polar=dict(
        radialaxis=dict(visible=True, range=[0, 1])
    ),
    showlegend=True,
    title='Comparação Normalizada (maior = melhor)'
)

st.plotly_chart(fig_radar, width='stretch')

# Feature Importance
st.markdown("---")
st.markdown("### 🔍 Importância das Features")

modelos_com_importance = [nome for nome, res in resultados.items() if 'feature_importance' in res]

if modelos_com_importance:
    modelo_fi = st.selectbox("Selecione o modelo:", modelos_com_importance)
    
    fi = resultados[modelo_fi]['feature_importance']
    df_fi = pd.DataFrame({
        'Feature': list(fi.keys()),
        'Importance': list(fi.values())
    }).sort_values('Importance', ascending=False).head(15)
    
    fig_fi = px.bar(
        df_fi,
        x='Importance',
        y='Feature',
        orientation='h',
        title=f'Top 15 Features - {modelo_fi}',
        color='Importance',
        color_continuous_scale='Blues'
    )
    fig_fi.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig_fi, width='stretch')
else:
    st.info("ℹ️ Nenhum modelo com feature importance disponível")

# Resumo
st.markdown("---")
st.markdown("### 📋 Resumo")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total de Modelos", len(resultados))

with col2:
    st.metric("Melhor MAE", f"{df_metrics['MAE'].min():.2f}")

with col3:
    st.metric("Melhor Modelo", best_model)
