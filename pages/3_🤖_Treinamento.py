"""
Página de Treinamento de Modelos - Com suporte a Turnos
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import logging
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Suprimir mensagens do cmdstanpy (usado pelo Prophet)
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
logging.getLogger('prophet').setLevel(logging.WARNING)

root_path = str(Path(__file__).parent.parent)
if root_path not in sys.path:
    sys.path.insert(0, root_path)

st.set_page_config(page_title="Treinamento", page_icon="🤖", layout="wide")

st.title("🤖 Treinamento de Modelos")

# Verificar se há dados
if 'df' not in st.session_state or st.session_state.df is None:
    st.warning("⚠️ Nenhum dado carregado. Vá para **Upload Dados** primeiro.")
    st.stop()

df_original = st.session_state.df.copy()

# Detectar colunas disponíveis
colunas_disponiveis = df_original.columns.tolist()
tem_turno = 'Turno' in colunas_disponiveis
tem_local = 'Local' in colunas_disponiveis
tem_sublocal = 'Sublocal' in colunas_disponiveis
tem_hora = 'Hora' in colunas_disponiveis

# Detectar granularidade dos dados
granularidade_tipo = st.session_state.get('granularidade_tipo', 'diaria')

st.markdown(f"**Dados carregados:** {len(df_original)} registros")

# Mostrar informações sobre granularidade
if tem_hora:
    horas_unicas = sorted(df_original['Hora'].dropna().unique().tolist())
    st.info(f"🕐 Granularidade: **Horária** | Horas: {min(horas_unicas):02d}h - {max(horas_unicas):02d}h")

if tem_turno:
    turnos_unicos = sorted(df_original['Turno'].dropna().unique().tolist())
    st.info(f"🕐 Turnos detectados: {', '.join(turnos_unicos)}")

# Filtros opcionais
df = df_original.copy()

# Opção de treinar por Local/Sublocal (e Hora)
treinar_por_local = False
treinar_por_hora = False

if tem_local or tem_sublocal or tem_hora:
    st.markdown("### 🏢 Modo de Agrupamento")
    
    opcoes_agrupamento = ["Agregado (todos juntos)"]
    
    if tem_hora:
        opcoes_agrupamento.append("Por Hora")
    if tem_local:
        opcoes_agrupamento.append("Por Local")
    if tem_local and tem_sublocal:
        opcoes_agrupamento.append("Por Local e Sublocal")
    if tem_hora and tem_local:
        opcoes_agrupamento.append("Por Hora e Local")
    if tem_hora and tem_local and tem_sublocal:
        opcoes_agrupamento.append("Por Hora, Local e Sublocal")
    
    modo_agrupamento = st.radio(
        "Como treinar os modelos?",
        options=opcoes_agrupamento,
        horizontal=True,
        help="Treina modelos separados para cada combinação selecionada"
    )
    
    treinar_por_local = 'Local' in modo_agrupamento
    treinar_por_hora = 'Hora' in modo_agrupamento
    
    # Mostrar resumo de grupos
    if modo_agrupamento != "Agregado (todos juntos)":
        if modo_agrupamento == "Por Hora":
            grupos = sorted(df['Hora'].dropna().unique().tolist())
            st.info(f"🕐 {len(grupos)} horas: {grupos[:5]}{'...' if len(grupos) > 5 else ''}")
        elif modo_agrupamento == "Por Local":
            grupos = df['Local'].dropna().unique().tolist()
            st.info(f"📍 {len(grupos)} locais")
        elif modo_agrupamento == "Por Local e Sublocal":
            grupos = df.groupby(['Local', 'Sublocal']).size().reset_index()
            st.info(f"📍 {len(grupos)} combinações Local/Sublocal")
        elif modo_agrupamento == "Por Hora e Local":
            grupos = df.groupby(['Hora', 'Local']).size().reset_index()
            st.info(f"🕐📍 {len(grupos)} combinações Hora/Local")
        elif modo_agrupamento == "Por Hora, Local e Sublocal":
            grupos = df.groupby(['Hora', 'Local', 'Sublocal']).size().reset_index()
            st.info(f"🕐📍 {len(grupos)} combinações Hora/Local/Sublocal")
else:
    modo_agrupamento = "Agregado (todos juntos)"

# Configurações
st.markdown("---")
st.markdown("### ⚙️ Configurações de Treinamento")

col1, col2, col3 = st.columns(3)

with col1:
    train_size = st.slider(
        "Tamanho do conjunto de treino (%)",
        min_value=50,
        max_value=95,
        value=80,
        step=5
    )

with col2:
    horizonte = st.number_input(
        "Horizonte de previsão (dias)",
        min_value=1,
        max_value=365,
        value=30
    )

with col3:
    # Modo de previsão
    if tem_turno:
        modo_previsao = st.radio(
            "Modo de Previsão",
            options=["Por Turno", "Agregado (Total)"],
            help="Por Turno: treina um modelo para cada turno. Agregado: soma todos os turnos."
        )
        treinar_por_turno = modo_previsao == "Por Turno"
    else:
        treinar_por_turno = False
        st.info("Previsão agregada (sem turnos)")

# Opções avançadas
st.markdown("### ⚙️ Configurações Avançadas")
col_adv1, col_adv2 = st.columns(2)

with col_adv1:
    validacao_walk_forward = st.checkbox(
        "🔄 Validação Walk-Forward",
        value=True,
        help="Usa validação mais realista que simula previsões sequenciais. Mais lento, mas evita overfitting."
    )

with col_adv2:
    usar_ensemble = st.checkbox(
        "🎯 Ensemble (Média Ponderada)",
        value=False,
        help="Combina previsões dos melhores modelos usando pesos baseados em performance."
    )

# Metas de performance (aparece quando ensemble está ativo)
if usar_ensemble:
    st.markdown("### 🎯 Metas de Performance")
    col_meta1, col_meta2, col_meta3 = st.columns(3)
    
    with col_meta1:
        meta_mape = st.number_input(
            "Meta MAPE (%)",
            min_value=1.0,
            max_value=50.0,
            value=10.0,
            step=1.0,
            help="O ensemble tentará atingir este MAPE combinando os melhores modelos"
        )
    
    with col_meta2:
        meta_mae = st.number_input(
            "Meta MAE",
            min_value=0.1,
            max_value=100.0,
            value=5.0,
            step=0.5,
            help="O ensemble tentará atingir este MAE"
        )
    
    with col_meta3:
        n_modelos_ensemble = st.slider(
            "Nº de modelos no ensemble",
            min_value=2,
            max_value=5,
            value=3,
            help="Quantos melhores modelos combinar"
        )
    
    metodo_ensemble = st.radio(
        "Método de combinação",
        options=["Média Ponderada por Score", "Média Ponderada por 1/MAPE", "Média Simples", "Mediana"],
        horizontal=True,
        help="Como combinar as previsões dos modelos"
    )
else:
    meta_mape = 10.0
    meta_mae = 5.0
    n_modelos_ensemble = 3
    metodo_ensemble = "Média Ponderada por Score"

# Seleção de modelos
st.markdown("### 🎯 Selecione os Modelos")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Modelos de ML**")
    use_xgboost = st.checkbox("XGBoost", value=True)
    use_lightgbm = st.checkbox("LightGBM", value=True)
    use_catboost = st.checkbox("CatBoost", value=False)
    use_rf = st.checkbox("Random Forest", value=True)
    use_gb = st.checkbox("Gradient Boosting", value=False)

with col2:
    st.markdown("**Redes Neurais / Regressão**")
    use_mlp = st.checkbox("MLP Regressor", value=False)
    use_ridge = st.checkbox("Ridge Regression", value=False)
    use_svr = st.checkbox("SVR (Support Vector)", value=False)
    use_elasticnet = st.checkbox("ElasticNet", value=False)
    use_knn = st.checkbox("KNN Regressor", value=False)

with col3:
    st.markdown("**Modelos Estatísticos**")
    use_prophet = st.checkbox("Prophet", value=True)
    use_arima = st.checkbox("ARIMA", value=False)
    use_sarima = st.checkbox("SARIMA", value=False)
    use_hw = st.checkbox("Holt-Winters", value=False)

# ========== FUNÇÕES AUXILIARES ==========

def create_features(df_input, apply_log=False):
    """Criar features temporais expandidas com todas as melhorias"""
    df_feat = df_input.copy()
    
    # ========== IMPORTANTE: Log-transform PRIMEIRO (antes das features de lag/rolling) ==========
    # Isso garante que todas as features baseadas em Demanda estejam na mesma escala
    if apply_log:
        df_feat['Demanda_original'] = df_feat['Demanda'].copy()
        df_feat['Demanda'] = np.log1p(df_feat['Demanda'])
    
    # ========== Features Temporais Básicas ==========
    df_feat['dayofweek'] = df_feat['Data'].dt.dayofweek
    df_feat['month'] = df_feat['Data'].dt.month
    df_feat['day'] = df_feat['Data'].dt.day
    df_feat['quarter'] = df_feat['Data'].dt.quarter
    df_feat['dayofyear'] = df_feat['Data'].dt.dayofyear
    df_feat['weekofyear'] = df_feat['Data'].dt.isocalendar().week.astype(int)
    df_feat['year'] = df_feat['Data'].dt.year
    df_feat['is_weekend'] = (df_feat['dayofweek'] >= 5).astype(int)
    df_feat['is_month_start'] = df_feat['Data'].dt.is_month_start.astype(int)
    df_feat['is_month_end'] = df_feat['Data'].dt.is_month_end.astype(int)
    
    # ========== Features Cíclicas (Seno/Cosseno) ==========
    df_feat['dayofweek_sin'] = np.sin(2 * np.pi * df_feat['dayofweek'] / 7)
    df_feat['dayofweek_cos'] = np.cos(2 * np.pi * df_feat['dayofweek'] / 7)
    df_feat['month_sin'] = np.sin(2 * np.pi * df_feat['month'] / 12)
    df_feat['month_cos'] = np.cos(2 * np.pi * df_feat['month'] / 12)
    df_feat['day_sin'] = np.sin(2 * np.pi * df_feat['day'] / 31)
    df_feat['day_cos'] = np.cos(2 * np.pi * df_feat['day'] / 31)
    df_feat['quarter_sin'] = np.sin(2 * np.pi * df_feat['quarter'] / 4)
    df_feat['quarter_cos'] = np.cos(2 * np.pi * df_feat['quarter'] / 4)
    
    # ========== Lags Estratégicos (agora na escala correta - log se aplicável) ==========
    for lag in [1, 2, 3, 7, 14, 21, 30]:
        df_feat[f'lag_{lag}'] = df_feat['Demanda'].shift(lag)
    
    # ========== Rolling Statistics (múltiplas janelas) ==========
    for window in [3, 7, 14, 30]:
        rolled = df_feat['Demanda'].shift(1).rolling(window=window, min_periods=1)
        df_feat[f'rolling_mean_{window}'] = rolled.mean()
        df_feat[f'rolling_std_{window}'] = rolled.std()
        df_feat[f'rolling_min_{window}'] = rolled.min()
        df_feat[f'rolling_max_{window}'] = rolled.max()
        df_feat[f'rolling_median_{window}'] = rolled.median()
        # Quartis
        df_feat[f'rolling_q25_{window}'] = rolled.quantile(0.25)
        df_feat[f'rolling_q75_{window}'] = rolled.quantile(0.75)
    
    # ========== Features de VOLATILIDADE (NOVO!) ==========
    # Coeficiente de Variação Rolling - detecta períodos de alta variação
    for window in [7, 14, 30]:
        rolling_mean = df_feat['Demanda'].shift(1).rolling(window=window, min_periods=2).mean()
        rolling_std = df_feat['Demanda'].shift(1).rolling(window=window, min_periods=2).std()
        df_feat[f'cv_{window}'] = (rolling_std / rolling_mean.replace(0, np.nan)).fillna(0)
        
        # Indicador de alta volatilidade (CV > 0.3)
        df_feat[f'high_volatility_{window}'] = (df_feat[f'cv_{window}'] > 0.3).astype(int)
    
    # ========== Range Interquartil (IQR) - Robustez a outliers ==========
    for window in [7, 14]:
        q75 = df_feat['Demanda'].shift(1).rolling(window=window, min_periods=2).quantile(0.75)
        q25 = df_feat['Demanda'].shift(1).rolling(window=window, min_periods=2).quantile(0.25)
        df_feat[f'iqr_{window}'] = q75 - q25
        
        # Normalizado pela mediana
        rolling_median = df_feat['Demanda'].shift(1).rolling(window=window, min_periods=2).median()
        df_feat[f'iqr_norm_{window}'] = (df_feat[f'iqr_{window}'] / rolling_median.replace(0, np.nan)).fillna(0)
    
    # ========== Mediana Móvel Exponencial (mais robusta que média) ==========
    df_feat['ewm_mean_7'] = df_feat['Demanda'].shift(1).ewm(span=7, adjust=False).mean()
    df_feat['ewm_mean_14'] = df_feat['Demanda'].shift(1).ewm(span=14, adjust=False).mean()
    df_feat['ewm_std_7'] = df_feat['Demanda'].shift(1).ewm(span=7, adjust=False).std()
    
    # ========== Desvio da Mediana (mais robusto que desvio da média) ==========
    for window in [7, 14]:
        rolling_median = df_feat['Demanda'].shift(1).rolling(window=window, min_periods=2).median()
        df_feat[f'deviation_from_median_{window}'] = df_feat['Demanda'].shift(1) - rolling_median
    
    # ========== Z-Score Robusto (usando MAD - Median Absolute Deviation) ==========
    for window in [7, 14]:
        rolling_median = df_feat['Demanda'].shift(1).rolling(window=window, min_periods=2).median()
        rolling_mad = df_feat['Demanda'].shift(1).rolling(window=window, min_periods=2).apply(
            lambda x: np.median(np.abs(x - np.median(x))), raw=True
        )
        df_feat[f'robust_zscore_{window}'] = ((df_feat['Demanda'].shift(1) - rolling_median) / 
                                               (1.4826 * rolling_mad.replace(0, np.nan))).fillna(0)
    
    # ========== Indicadores de Anomalia ==========
    # Detectar se valor anterior estava fora de 2 desvios padrão
    rolling_mean_14 = df_feat['Demanda'].shift(1).rolling(window=14, min_periods=2).mean()
    rolling_std_14 = df_feat['Demanda'].shift(1).rolling(window=14, min_periods=2).std()
    upper_bound = rolling_mean_14 + 2 * rolling_std_14
    lower_bound = rolling_mean_14 - 2 * rolling_std_14
    df_feat['was_anomaly'] = ((df_feat['Demanda'].shift(1) > upper_bound) | 
                              (df_feat['Demanda'].shift(1) < lower_bound)).astype(int)
    
    # ========== CORREÇÃO DE VIÉS: Features para detectar valores inflacionados ==========
    # Razão entre média recente e média de longo prazo (detecta se recente está alto)
    media_curta = df_feat['Demanda'].shift(1).rolling(window=7, min_periods=1).mean()
    media_longa = df_feat['Demanda'].shift(1).rolling(window=30, min_periods=7).mean()
    df_feat['ratio_short_long'] = (media_curta / media_longa.replace(0, 1)).fillna(1)
    
    # Indicador: valores recentes estão acima da média histórica
    df_feat['above_long_avg'] = (media_curta > media_longa).astype(int)
    
    # Distância percentual da mediana (mais robusto)
    mediana_30 = df_feat['Demanda'].shift(1).rolling(window=30, min_periods=7).median()
    df_feat['pct_from_median_30'] = ((df_feat['Demanda'].shift(1) - mediana_30) / mediana_30.replace(0, 1)).fillna(0)
    
    # ========== Tendências Locais (Regressão Linear) ==========
    for window in [7, 14, 30]:
        df_feat[f'trend_{window}'] = df_feat['Demanda'].shift(1).rolling(window=window, min_periods=2).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=False
        )
    
    # ========== Features de Interação ==========
    df_feat['weekend_month'] = df_feat['is_weekend'] * df_feat['month']
    df_feat['dayofweek_month'] = df_feat['dayofweek'] * df_feat['month']
    df_feat['dayofweek_quarter'] = df_feat['dayofweek'] * df_feat['quarter']
    
    # ========== Interação Volatilidade x Dia da Semana ==========
    df_feat['volatility_weekend'] = df_feat['cv_7'] * df_feat['is_weekend']
    df_feat['volatility_dayofweek'] = df_feat['cv_7'] * df_feat['dayofweek']
    
    # ========== Diferenças (agora na escala log se aplicável) ==========
    df_feat['diff_1'] = df_feat['Demanda'].diff(1)
    df_feat['diff_7'] = df_feat['Demanda'].diff(7)
    
    # ========== Variação percentual ==========
    df_feat['pct_change_1'] = df_feat['Demanda'].pct_change(1)
    df_feat['pct_change_7'] = df_feat['Demanda'].pct_change(7)
    
    # Log-transform já foi aplicado no início da função (antes dos lags/rolling)
    
    return df_feat

def calculate_cv(y):
    """Calcular Coeficiente de Variação"""
    mean_y = np.mean(y)
    if mean_y == 0:
        return 0
    return np.std(y) / mean_y

def calculate_smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error"""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denominator != 0
    if mask.sum() == 0:
        return 0
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask])

def calculate_mase(y_true, y_pred, y_train, seasonality=7):
    """Mean Absolute Scaled Error - compara com baseline naive sazonal"""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    y_train = np.array(y_train).flatten()
    
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Naive sazonal baseline
    if len(y_train) > seasonality:
        naive_errors = np.abs(y_train[seasonality:] - y_train[:-seasonality])
        naive_mae = np.mean(naive_errors)
    else:
        naive_mae = np.std(y_train)
    
    if naive_mae == 0:
        return mae
    return mae / naive_mae

def calculate_metrics(y_true, y_pred, y_train=None):
    """Calcular métricas incluindo SMAPE e MASE"""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # MAPE
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))
    else:
        mape = 0
    
    # R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # SMAPE
    smape = calculate_smape(y_true, y_pred)
    
    # MASE
    if y_train is not None:
        mase = calculate_mase(y_true, y_pred, y_train)
    else:
        mase = mae / (np.std(y_true) if np.std(y_true) > 0 else 1)
    
    return {'mae': mae, 'rmse': rmse, 'mape': mape, 'r2': r2, 'smape': smape, 'mase': mase}

def criar_ensemble(resultados, n_modelos=3, metodo='Média Ponderada por Score', 
                   meta_mape=10.0, meta_mae=5.0):
    """
    Cria ensemble combinando os melhores modelos para atingir metas de MAPE/MAE.
    Usa predictions e y_test já salvos nos resultados.
    
    Returns:
        dict com modelo ensemble, previsões e métricas
    """
    if len(resultados) < 2:
        return None
    
    # Ordenar modelos por score (maior = melhor)
    modelos_ordenados = []
    for nome, res in resultados.items():
        if 'model' not in res or nome == '🎯 Ensemble':
            continue
        if 'predictions' not in res or 'y_test' not in res:
            continue
        m = res['metrics']
        mape_val = max(m['mape'], 0.001)
        smape_val = max(m.get('smape', 0.1), 0.001)
        mase_val = max(m.get('mase', 1), 0.001)
        score = (1/mape_val + 1/smape_val + 1/mase_val)
        modelos_ordenados.append({
            'nome': nome,
            'model': res['model'],
            'metrics': m,
            'score': score,
            'predictions': res.get('predictions', None),
            'y_test': res.get('y_test', None),
        })
    
    modelos_ordenados.sort(key=lambda x: x['score'], reverse=True)
    
    # Selecionar top N modelos
    top_modelos = modelos_ordenados[:min(n_modelos, len(modelos_ordenados))]
    
    if len(top_modelos) < 2:
        return None
    
    # Obter y_test de referência (todos devem ser iguais)
    y_test_ref = None
    for m in top_modelos:
        if m['y_test'] is not None and hasattr(m['y_test'], '__len__'):
            y_test_ref = np.array(m['y_test'])
            break
    
    if y_test_ref is None:
        return None
    
    # Coletar previsões de cada modelo
    todas_predicoes = []
    pesos = []
    nomes_modelos = []
    
    for m in top_modelos:
        predictions = m.get('predictions', None)
        if predictions is not None and hasattr(predictions, '__len__') and len(predictions) > 0:
            todas_predicoes.append(np.array(predictions))
            
            if metodo == 'Média Ponderada por Score':
                pesos.append(m['score'])
            elif metodo == 'Média Ponderada por 1/MAPE':
                pesos.append(1 / max(m['metrics']['mape'], 0.001))
            else:
                pesos.append(1.0)
            
            nomes_modelos.append(m['nome'])
    
    if len(todas_predicoes) < 2:
        return None
    
    # Normalizar pesos
    pesos = np.array(pesos)
    pesos = pesos / pesos.sum()
    
    # Calcular ensemble
    min_len = min(len(p) for p in todas_predicoes)
    min_len = min(min_len, len(y_test_ref))
    todas_predicoes = [p[:min_len] for p in todas_predicoes]
    
    if metodo == 'Mediana':
        ensemble_pred = np.median(np.array(todas_predicoes), axis=0)
    else:
        # Média ponderada
        ensemble_pred = np.zeros(min_len)
        for i, pred in enumerate(todas_predicoes):
            ensemble_pred += pesos[i] * pred
    
    # Clipping não-negativo
    ensemble_pred = np.maximum(ensemble_pred, 0)
    
    # Calcular métricas do ensemble
    y_test_cut = y_test_ref[:min_len]
    ensemble_metrics = calculate_metrics(y_test_cut, ensemble_pred)
    
    # Verificar se atingiu metas
    atingiu_mape = ensemble_metrics['mape'] * 100 <= meta_mape
    atingiu_mae = ensemble_metrics['mae'] <= meta_mae
    
    return {
        'y_pred': ensemble_pred,
        'y_true': y_test_cut,
        'mae': ensemble_metrics['mae'],
        'mape': ensemble_metrics['mape'] * 100,  # Já em %
        'smape': ensemble_metrics['smape'] * 100,
        'mase': ensemble_metrics['mase'],
        'r2': ensemble_metrics['r2'],
        'modelos': nomes_modelos,
        'pesos': pesos.tolist(),
        'metodo': metodo,
        'atingiu_mape': atingiu_mape,
        'atingiu_mae': atingiu_mae,
    }

def otimizar_ensemble_para_meta(resultados, meta_mape=10.0, meta_mae=5.0, 
                                 n_modelos=3, metodo_preferido='Média Ponderada por Score'):
    """
    Tenta diferentes combinações de modelos para atingir as metas.
    """
    from itertools import combinations
    
    modelos_disponiveis = [nome for nome, res in resultados.items() 
                          if 'model' in res and 'predictions' in res and 'y_test' in res
                          and nome != '🎯 Ensemble']
    
    if len(modelos_disponiveis) < 2:
        return None
    
    melhor_ensemble = None
    melhor_score = -float('inf')
    
    # Testar diferentes combinações
    for n in range(2, min(n_modelos + 1, len(modelos_disponiveis) + 1)):
        for combo in combinations(modelos_disponiveis, n):
            # Criar sub-resultados
            sub_resultados = {nome: resultados[nome] for nome in combo}
            
            # Se tiver método preferido, testar ele primeiro
            metodos = [metodo_preferido] if metodo_preferido != 'Média Simples' else ['Média Ponderada por Score']
            if 'Média Ponderada por Score' not in metodos:
                metodos.append('Média Ponderada por Score')
            metodos.extend(['Média Ponderada por 1/MAPE', 'Mediana'])
            metodos = list(dict.fromkeys(metodos))  # Remover duplicatas mantendo ordem
            
            for metodo in metodos:
                ensemble = criar_ensemble(
                    sub_resultados,
                    n_modelos=n, 
                    metodo=metodo,
                    meta_mape=meta_mape, 
                    meta_mae=meta_mae
                )
                
                if ensemble is None:
                    continue
                
                # Calcular score baseado em quão perto está das metas
                mape_atual = ensemble['mape']
                mae_atual = ensemble['mae']
                
                # Score: quanto menor MAPE e MAE, melhor
                # Bonus se atingir metas
                score = -mape_atual - mae_atual
                if mape_atual <= meta_mape:
                    score += 50
                if mae_atual <= meta_mae:
                    score += 50
                
                if score > melhor_score:
                    melhor_score = score
                    melhor_ensemble = ensemble
                    melhor_ensemble['combinacao'] = list(combo)
                    
                    # Se atingiu ambas metas, pode parar
                    if ensemble['atingiu_mape'] and ensemble['atingiu_mae']:
                        return melhor_ensemble
    
    return melhor_ensemble

def walk_forward_validation(model, X, y, n_splits=5, test_size=None):
    """
    Validação Walk-Forward mais realista para séries temporais.
    Treina em dados crescentes e testa em próximo período.
    """
    n = len(X)
    if test_size is None:
        test_size = max(1, n // (n_splits + 1))
    
    initial_train_size = n - (n_splits * test_size)
    if initial_train_size < 10:
        initial_train_size = 10
        n_splits = max(1, (n - initial_train_size) // test_size)
    
    all_preds = []
    all_trues = []
    
    for i in range(n_splits):
        train_end = initial_train_size + i * test_size
        test_start = train_end
        test_end = min(test_start + test_size, n)
        
        if test_start >= n:
            break
        
        X_train_wf = X[:train_end]
        y_train_wf = y[:train_end]
        X_test_wf = X[test_start:test_end]
        y_test_wf = y[test_start:test_end]
        
        if len(X_train_wf) < 5 or len(X_test_wf) < 1:
            continue
        
        try:
            # Clone do modelo para não afetar o original
            from sklearn.base import clone
            model_clone = clone(model)
            model_clone.fit(X_train_wf, y_train_wf)
            preds = model_clone.predict(X_test_wf)
            
            all_preds.extend(preds)
            all_trues.extend(y_test_wf)
        except:
            continue
    
    if len(all_preds) == 0:
        return None, None
    
    return np.array(all_trues), np.array(all_preds)

def treinar_modelos(df_turno, turno_nome, modelos_selecionados, train_size, usar_walk_forward=True):
    """Treinar modelos para um turno específico ou agregado"""
    from sklearn.preprocessing import RobustScaler
    from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
    
    resultados = {}
    
    # Verificar se há múltiplos registros por dia
    df_check = df_turno.groupby('Data').size()
    tem_multiplos = df_check.max() > 1
    
    if tem_multiplos:
        # Usar método de agregação salvo ou soma como padrão
        metodo_agg = st.session_state.get('metodo_agregacao', 'sum')
        if metodo_agg is None:
            metodo_agg = 'sum'
        
        if metodo_agg == 'p80':
            df_agg = df_turno.groupby('Data').agg({'Demanda': lambda x: x.quantile(0.8)}).reset_index()
        else:
            df_agg = df_turno.groupby('Data').agg({'Demanda': metodo_agg}).reset_index()
    else:
        # Dados já estão agregados por dia
        df_agg = df_turno.groupby('Data').agg({'Demanda': 'sum'}).reset_index()
    
    df_agg = df_agg.sort_values('Data').reset_index(drop=True)
    
    if len(df_agg) < 10:
        return None, f"Dados insuficientes para {turno_nome} ({len(df_agg)} dias)"
    
    # ========== Análise de Volatilidade ==========
    cv = calculate_cv(df_agg['Demanda'].values)
    std_demanda = df_agg['Demanda'].std()
    mean_demanda = df_agg['Demanda'].mean()
    
    # Detectar nível de volatilidade
    # Alta volatilidade: CV > 0.5 OU std > 5
    alta_volatilidade = cv > 0.5 or std_demanda > 5
    muito_alta_volatilidade = cv > 0.8 or std_demanda > 10
    
    # Log-transform condicional
    apply_log = cv > 0.5
    
    # Mostrar info de volatilidade
    if muito_alta_volatilidade:
        st.warning(f"⚠️ **{turno_nome}**: Volatilidade MUITO ALTA (CV={cv:.2f}, std={std_demanda:.2f}). Usando configurações robustas.")
    elif alta_volatilidade:
        st.info(f"📊 **{turno_nome}**: Volatilidade alta (CV={cv:.2f}, std={std_demanda:.2f}). Aplicando regularização extra.")
    
    # Criar features no dataset completo
    df_full_feat = create_features(df_agg, apply_log=apply_log)
    
    # Preencher NaN
    for col in df_full_feat.columns:
        if df_full_feat[col].isna().any():
            fill_value = df_full_feat[col].mean() if df_full_feat[col].notna().any() else 0
            df_full_feat[col] = df_full_feat[col].fillna(fill_value)
    
    # Substituir inf por valores grandes mas finitos
    df_full_feat = df_full_feat.replace([np.inf, -np.inf], 0)
    
    # Split treino/teste
    split_idx = int(len(df_full_feat) * train_size / 100)
    split_idx = max(split_idx, min(10, len(df_full_feat) - 3))
    
    df_train_feat = df_full_feat.iloc[:split_idx].copy()
    df_test_feat = df_full_feat.iloc[split_idx:].copy()
    
    if len(df_test_feat) < 3:
        n_val = max(3, int(len(df_train_feat) * 0.2))
        df_test_feat = df_train_feat.tail(n_val).copy()
        df_train_feat = df_train_feat.iloc[:-n_val].copy()
    
    # Dados originais para modelos estatísticos
    df_train = df_agg.iloc[:split_idx].copy()
    df_test = df_agg.iloc[split_idx:].copy()
    if len(df_test) < 3:
        n_val = max(3, int(len(df_train) * 0.2))
        df_test = df_train.tail(n_val).copy()
        df_train = df_train.iloc[:-n_val].copy()
    
    # Preparar features
    feature_cols = [col for col in df_train_feat.columns 
                   if col not in ['Data', 'Demanda', 'Demanda_original', 'Turno', 'Local', 'Sublocal', 'ds', 'y']
                   and pd.api.types.is_numeric_dtype(df_train_feat[col])]
    
    X_train = np.nan_to_num(df_train_feat[feature_cols].values.astype(np.float64), nan=0.0)
    y_train = df_train_feat['Demanda'].values.astype(np.float64)
    X_test = np.nan_to_num(df_test_feat[feature_cols].values.astype(np.float64), nan=0.0)
    y_test = df_test_feat['Demanda'].values.astype(np.float64)
    
    # y_train original para MASE
    y_train_original = df_train['Demanda'].values.astype(np.float64)
    y_test_original = df_test['Demanda'].values.astype(np.float64)
    
    if len(X_train) == 0:
        return None, f"Dados de treino vazios para {turno_nome}"
    
    # ========== RobustScaler ==========
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # TimeSeriesSplit para cross-validation
    n_splits = min(5, len(X_train) // 10) if len(X_train) >= 50 else min(3, len(X_train) // 5)
    n_splits = max(2, n_splits)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Treinar cada modelo
    for modelo_nome in modelos_selecionados:
        try:
            if modelo_nome == 'XGBoost':
                from xgboost import XGBRegressor
                
                # Parâmetros otimizados e mais conservadores
                if len(X_train) >= 50:
                    # GridSearch simplificado para ser mais rápido
                    param_grid = {
                        'n_estimators': [50, 100],
                        'max_depth': [3, 5],
                        'learning_rate': [0.05, 0.1],
                        'reg_lambda': [1, 10] if not alta_volatilidade else [10, 50, 100]
                    }
                    
                    base_model = XGBRegressor(
                        random_state=42, verbosity=0,
                        subsample=0.7 if alta_volatilidade else 0.8, 
                        colsample_bytree=0.7 if alta_volatilidade else 0.8,
                        min_child_weight=3 if alta_volatilidade else 1
                    )
                    
                    grid_search = GridSearchCV(
                        base_model, param_grid, cv=tscv, 
                        scoring='neg_mean_absolute_error',
                        n_jobs=-1, refit=True
                    )
                    grid_search.fit(X_train_scaled, y_train)
                    model = grid_search.best_estimator_
                else:
                    # Configuração com regularização forte para alta volatilidade
                    reg_lambda = 50 if muito_alta_volatilidade else (20 if alta_volatilidade else 5)
                    model = XGBRegressor(
                        n_estimators=50, max_depth=3, learning_rate=0.05 if alta_volatilidade else 0.1,
                        subsample=0.7 if alta_volatilidade else 0.8, 
                        colsample_bytree=0.7 if alta_volatilidade else 0.8, 
                        reg_lambda=reg_lambda,
                        min_child_weight=3 if alta_volatilidade else 1,
                        random_state=42, verbosity=0
                    )
                    model.fit(X_train_scaled, y_train)
                
                # Calcular métricas usando walk-forward se ativado
                if usar_walk_forward and len(X_train_scaled) >= 20:
                    y_true_wf, y_pred_wf = walk_forward_validation(
                        model, X_train_scaled, y_train, n_splits=min(5, len(X_train)//10)
                    )
                    if y_true_wf is not None:
                        if apply_log:
                            y_pred_wf = np.expm1(y_pred_wf)
                            y_true_wf_orig = np.expm1(y_true_wf)
                        else:
                            y_true_wf_orig = y_true_wf
                        y_pred_wf = np.maximum(y_pred_wf, 0)
                        metrics_wf = calculate_metrics(y_true_wf_orig, y_pred_wf, y_train_original)
                    else:
                        metrics_wf = None
                else:
                    metrics_wf = None
                
                # Retreinar no dataset completo para usar na previsão
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                if apply_log:
                    y_pred = np.expm1(y_pred)
                y_pred = np.maximum(y_pred, 0)
                
                # Usar métricas walk-forward se disponíveis (mais realistas)
                if metrics_wf:
                    metrics_final = metrics_wf
                else:
                    metrics_final = calculate_metrics(y_test_original, y_pred, y_train_original)
                
                resultados[modelo_nome] = {
                    'model': model,
                    'scaler': scaler,
                    'predictions': y_pred,
                    'y_test': y_test_original,
                    'metrics': metrics_final,
                    'feature_cols': feature_cols,
                    'apply_log': apply_log
                }
            
            elif modelo_nome == 'LightGBM':
                from lightgbm import LGBMRegressor
                
                if len(X_train) >= 50:
                    param_grid = {
                        'n_estimators': [50, 100],
                        'max_depth': [3, 4] if alta_volatilidade else [3, 5],
                        'learning_rate': [0.03, 0.05] if alta_volatilidade else [0.05, 0.1],
                        'reg_lambda': [10, 50, 100] if alta_volatilidade else [1, 10]
                    }
                    
                    base_model = LGBMRegressor(
                        random_state=42, verbosity=-1,
                        subsample=0.7 if alta_volatilidade else 0.8, 
                        colsample_bytree=0.7 if alta_volatilidade else 0.8,
                        min_child_samples=10 if alta_volatilidade else 5
                    )
                    
                    grid_search = GridSearchCV(
                        base_model, param_grid, cv=tscv,
                        scoring='neg_mean_absolute_error',
                        n_jobs=-1, refit=True
                    )
                    grid_search.fit(X_train_scaled, y_train)
                    model = grid_search.best_estimator_
                else:
                    reg_lambda = 50 if muito_alta_volatilidade else (20 if alta_volatilidade else 5)
                    model = LGBMRegressor(
                        n_estimators=50, max_depth=3, 
                        learning_rate=0.03 if alta_volatilidade else 0.1,
                        subsample=0.7 if alta_volatilidade else 0.8, 
                        colsample_bytree=0.7 if alta_volatilidade else 0.8, 
                        reg_lambda=reg_lambda,
                        min_child_samples=10 if alta_volatilidade else 5,
                        random_state=42, verbosity=-1
                    )
                    model.fit(X_train_scaled, y_train)
                
                # Walk-forward validation
                if usar_walk_forward and len(X_train_scaled) >= 20:
                    y_true_wf, y_pred_wf = walk_forward_validation(
                        model, X_train_scaled, y_train, n_splits=min(5, len(X_train)//10)
                    )
                    if y_true_wf is not None:
                        if apply_log:
                            y_pred_wf = np.expm1(y_pred_wf)
                            y_true_wf_orig = np.expm1(y_true_wf)
                        else:
                            y_true_wf_orig = y_true_wf
                        y_pred_wf = np.maximum(y_pred_wf, 0)
                        metrics_wf = calculate_metrics(y_true_wf_orig, y_pred_wf, y_train_original)
                    else:
                        metrics_wf = None
                else:
                    metrics_wf = None
                
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                if apply_log:
                    y_pred = np.expm1(y_pred)
                y_pred = np.maximum(y_pred, 0)
                
                if metrics_wf:
                    metrics_final = metrics_wf
                else:
                    metrics_final = calculate_metrics(y_test_original, y_pred, y_train_original)
                
                resultados[modelo_nome] = {
                    'model': model,
                    'scaler': scaler,
                    'predictions': y_pred,
                    'y_test': y_test_original,
                    'metrics': metrics_final,
                    'feature_cols': feature_cols,
                    'apply_log': apply_log
                }
            
            elif modelo_nome == 'CatBoost':
                from catboost import CatBoostRegressor
                model = CatBoostRegressor(
                    iterations=100, depth=4, learning_rate=0.1,
                    l2_leaf_reg=5, random_state=42, verbose=0
                )
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                if apply_log:
                    y_pred = np.expm1(y_pred)
                y_pred = np.maximum(y_pred, 0)
                
                resultados[modelo_nome] = {
                    'model': model,
                    'scaler': scaler,
                    'predictions': y_pred,
                    'y_test': y_test_original,
                    'metrics': calculate_metrics(y_test_original, y_pred, y_train_original),
                    'feature_cols': feature_cols,
                    'apply_log': apply_log
                }
            
            elif modelo_nome == 'RandomForest':
                from sklearn.ensemble import RandomForestRegressor
                
                model = RandomForestRegressor(
                    n_estimators=100, max_depth=8, min_samples_split=5,
                    min_samples_leaf=3, random_state=42, n_jobs=-1
                )
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                if apply_log:
                    y_pred = np.expm1(y_pred)
                y_pred = np.maximum(y_pred, 0)
                
                resultados[modelo_nome] = {
                    'model': model,
                    'scaler': scaler,
                    'predictions': y_pred,
                    'y_test': y_test_original,
                    'metrics': calculate_metrics(y_test_original, y_pred, y_train_original),
                    'feature_cols': feature_cols,
                    'apply_log': apply_log
                }
            
            elif modelo_nome == 'GradientBoosting':
                from sklearn.ensemble import GradientBoostingRegressor
                model = GradientBoostingRegressor(
                    n_estimators=100, max_depth=4, learning_rate=0.1,
                    subsample=0.8, random_state=42
                )
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                if apply_log:
                    y_pred = np.expm1(y_pred)
                y_pred = np.maximum(y_pred, 0)
                
                resultados[modelo_nome] = {
                    'model': model,
                    'scaler': scaler,
                    'predictions': y_pred,
                    'y_test': y_test_original,
                    'metrics': calculate_metrics(y_test_original, y_pred, y_train_original),
                    'feature_cols': feature_cols,
                    'apply_log': apply_log
                }
            
            elif modelo_nome == 'MLPRegressor':
                from sklearn.neural_network import MLPRegressor
                
                # Configuração mais robusta para alta volatilidade
                alpha = 0.1 if muito_alta_volatilidade else (0.05 if alta_volatilidade else 0.01)
                model = MLPRegressor(
                    hidden_layer_sizes=(32, 16) if alta_volatilidade else (64, 32),
                    alpha=alpha,  # Regularização L2 mais forte
                    learning_rate_init=0.005 if alta_volatilidade else 0.01,
                    max_iter=500,
                    early_stopping=True,
                    validation_fraction=0.2 if alta_volatilidade else 0.15,
                    random_state=42
                )
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                if apply_log:
                    y_pred = np.expm1(y_pred)
                y_pred = np.maximum(y_pred, 0)
                
                resultados[modelo_nome] = {
                    'model': model,
                    'scaler': scaler,
                    'predictions': y_pred,
                    'y_test': y_test_original,
                    'metrics': calculate_metrics(y_test_original, y_pred, y_train_original),
                    'feature_cols': feature_cols,
                    'apply_log': apply_log
                }
            
            elif modelo_nome == 'Ridge':
                from sklearn.linear_model import Ridge
                
                # Regularização muito mais forte para alta volatilidade
                alpha = 100.0 if muito_alta_volatilidade else (50.0 if alta_volatilidade else 10.0)
                model = Ridge(alpha=alpha)
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                if apply_log:
                    y_pred = np.expm1(y_pred)
                y_pred = np.maximum(y_pred, 0)
                
                resultados[modelo_nome] = {
                    'model': model,
                    'scaler': scaler,
                    'predictions': y_pred,
                    'y_test': y_test_original,
                    'metrics': calculate_metrics(y_test_original, y_pred, y_train_original),
                    'feature_cols': feature_cols,
                    'apply_log': apply_log
                }
            
            elif modelo_nome == 'SVR':
                from sklearn.svm import SVR
                
                model = SVR(C=1.0, epsilon=0.1, kernel='rbf')
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                if apply_log:
                    y_pred = np.expm1(y_pred)
                y_pred = np.maximum(y_pred, 0)
                
                resultados[modelo_nome] = {
                    'model': model,
                    'scaler': scaler,
                    'predictions': y_pred,
                    'y_test': y_test_original,
                    'metrics': calculate_metrics(y_test_original, y_pred, y_train_original),
                    'feature_cols': feature_cols,
                    'apply_log': apply_log
                }
            
            elif modelo_nome == 'ElasticNet':
                from sklearn.linear_model import ElasticNet
                
                model = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=1000, random_state=42)
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                if apply_log:
                    y_pred = np.expm1(y_pred)
                y_pred = np.maximum(y_pred, 0)
                
                resultados[modelo_nome] = {
                    'model': model,
                    'scaler': scaler,
                    'predictions': y_pred,
                    'y_test': y_test_original,
                    'metrics': calculate_metrics(y_test_original, y_pred, y_train_original),
                    'feature_cols': feature_cols,
                    'apply_log': apply_log
                }
            
            elif modelo_nome == 'Prophet':
                from prophet import Prophet
                from prophet.diagnostics import cross_validation, performance_metrics
                
                df_prophet = df_train[['Data', 'Demanda']].copy()
                df_prophet.columns = ['ds', 'y']
                
                # ========== Prophet OTIMIZADO para Alta Volatilidade ==========
                n_dias = len(df_prophet)
                media_demanda = df_prophet['y'].mean()
                std_demanda = df_prophet['y'].std()
                cv_demanda = std_demanda / media_demanda if media_demanda > 0 else 0
                
                # NOVA CONFIGURAÇÃO: Muito mais conservadora para alta volatilidade
                if muito_alta_volatilidade:
                    # Configuração ultra-conservadora
                    changepoint_prior = 0.001  # Quase sem mudanças de tendência
                    seasonality_prior = 0.1    # Sazonalidade muito suave
                    seasonality_mode = 'additive'
                    growth = 'flat'  # Sem tendência
                    n_changepoints = 5
                elif alta_volatilidade:
                    # Configuração conservadora
                    changepoint_prior = 0.01
                    seasonality_prior = 1.0
                    seasonality_mode = 'additive'
                    growth = 'linear'
                    n_changepoints = 10
                else:
                    # Configuração normal
                    changepoint_prior = 0.05
                    seasonality_prior = 10
                    seasonality_mode = 'multiplicative' if cv_demanda > 0.3 else 'additive'
                    growth = 'linear'
                    n_changepoints = 25
                
                # Criar modelo com parâmetros otimizados
                model = Prophet(
                    yearly_seasonality=False,  # Desativar - geralmente não ajuda com poucos dados
                    weekly_seasonality=True,
                    daily_seasonality=False,
                    seasonality_mode=seasonality_mode,
                    changepoint_prior_scale=changepoint_prior,
                    seasonality_prior_scale=seasonality_prior,
                    holidays_prior_scale=0.1 if alta_volatilidade else 10,
                    interval_width=0.80,  # Intervalo mais estreito
                    growth=growth,
                    n_changepoints=n_changepoints,
                    changepoint_range=0.8
                )
                
                # Adicionar sazonalidade semanal customizada (mais suave)
                model.add_seasonality(
                    name='weekly_custom', 
                    period=7, 
                    fourier_order=2 if alta_volatilidade else 3,
                    prior_scale=0.5 if alta_volatilidade else 5
                )
                
                # Adicionar regressor de fim de semana
                df_prophet['is_weekend'] = df_prophet['ds'].dt.dayofweek.isin([5, 6]).astype(float)
                model.add_regressor('is_weekend', prior_scale=0.5 if alta_volatilidade else 5)
                
                # Adicionar regressor de dia da semana (one-hot simplificado)
                df_prophet['is_monday'] = (df_prophet['ds'].dt.dayofweek == 0).astype(float)
                df_prophet['is_friday'] = (df_prophet['ds'].dt.dayofweek == 4).astype(float)
                model.add_regressor('is_monday', prior_scale=0.5 if alta_volatilidade else 2)
                model.add_regressor('is_friday', prior_scale=0.5 if alta_volatilidade else 2)
                
                model.fit(df_prophet, seed=42)
                
                # Preparar dados de teste
                future = df_test[['Data']].copy()
                future.columns = ['ds']
                future['is_weekend'] = future['ds'].dt.dayofweek.isin([5, 6]).astype(float)
                future['is_monday'] = (future['ds'].dt.dayofweek == 0).astype(float)
                future['is_friday'] = (future['ds'].dt.dayofweek == 4).astype(float)
                
                forecast = model.predict(future)
                y_pred = forecast['yhat'].values
                
                # Clipping mais agressivo para alta volatilidade
                if alta_volatilidade:
                    # Usar percentis ao invés de min/max
                    y_min = max(0, df_prophet['y'].quantile(0.05))
                    y_max = df_prophet['y'].quantile(0.95) * 1.2
                else:
                    y_min = max(0, df_prophet['y'].min() * 0.5)
                    y_max = df_prophet['y'].max() * 1.5
                y_pred = np.clip(y_pred, y_min, y_max)
                
                # Suavização adicional: misturar com média móvel se alta volatilidade
                if alta_volatilidade and len(df_prophet) >= 14:
                    media_historica = df_prophet['y'].tail(14).mean()
                    # Blend: 70% Prophet, 30% média histórica
                    y_pred = 0.7 * y_pred + 0.3 * media_historica
                
                y_test_prophet = df_test['Demanda'].values
                min_len = min(len(y_test_prophet), len(y_pred))
                
                resultados[modelo_nome] = {
                    'model': model,
                    'predictions': y_pred[:min_len],
                    'y_test': y_test_prophet[:min_len],
                    'metrics': calculate_metrics(y_test_prophet[:min_len], y_pred[:min_len], y_train_original),
                    'seasonality_mode': seasonality_mode
                }
            
            elif modelo_nome == 'ARIMA':
                from statsmodels.tsa.arima.model import ARIMA
                import warnings
                y_train_arima = df_train['Demanda'].values
                y_test_arima = df_test['Demanda'].values
                
                # Calcular volatilidade para adaptar ordem
                cv_arima = np.std(y_train_arima) / np.mean(y_train_arima) if np.mean(y_train_arima) > 0 else 0
                
                # Ordens adaptativas: menos diferenciação para alta volatilidade
                if cv_arima > 0.5:
                    orders = [(1, 0, 0), (1, 0, 1), (2, 0, 1)]  # Sem diferenciação
                elif cv_arima > 0.3:
                    orders = [(1, 1, 0), (2, 1, 1), (1, 1, 1)]  # Diferenciação simples
                else:
                    orders = [(5, 1, 0), (2, 1, 2), (1, 1, 1), (3, 1, 0)]
                
                model_fit = None
                best_aic = float('inf')
                best_model = None
                
                for order in orders:
                    try:
                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore')
                            model = ARIMA(y_train_arima, order=order)
                            fit = model.fit(method_kwargs={'maxiter': 500, 'disp': False})
                            if fit.aic < best_aic:
                                best_aic = fit.aic
                                best_model = fit
                    except:
                        continue
                
                if best_model:
                    y_pred = best_model.forecast(steps=len(y_test_arima))
                    # Clipping adaptativo
                    if cv_arima > 0.3:
                        y_min = max(0, np.percentile(y_train_arima, 5))
                        y_max = np.percentile(y_train_arima, 95) * 1.3
                        y_pred = np.clip(y_pred, y_min, y_max)
                    resultados[modelo_nome] = {
                        'model': best_model,
                        'predictions': y_pred,
                        'y_test': y_test_arima,
                        'metrics': calculate_metrics(y_test_arima, y_pred, y_train_original)
                    }
            
            elif modelo_nome == 'SARIMA':
                from statsmodels.tsa.statespace.sarimax import SARIMAX
                import warnings
                y_train_sarima = df_train['Demanda'].values
                y_test_sarima = df_test['Demanda'].values
                
                cv_sarima = np.std(y_train_sarima) / np.mean(y_train_sarima) if np.mean(y_train_sarima) > 0 else 0
                
                # Configurações de SARIMA (p,d,q)(P,D,Q,s)
                # s=7 para sazonalidade semanal
                if cv_sarima > 0.5:
                    # Alta volatilidade: sem diferenciação, sem sazonalidade complexa
                    seasonal_configs = [
                        ((1, 0, 0), (0, 0, 0, 7)),
                        ((1, 0, 1), (0, 0, 0, 7)),
                        ((2, 0, 1), (0, 0, 0, 7)),
                    ]
                elif cv_sarima > 0.3:
                    seasonal_configs = [
                        ((1, 1, 0), (0, 1, 0, 7)),
                        ((1, 1, 1), (0, 0, 1, 7)),
                        ((1, 0, 1), (1, 0, 1, 7)),
                    ]
                else:
                    seasonal_configs = [
                        ((1, 1, 1), (1, 1, 1, 7)),
                        ((1, 1, 1), (0, 1, 1, 7)),
                        ((2, 1, 1), (1, 0, 1, 7)),
                        ((1, 1, 0), (1, 1, 0, 7)),
                    ]
                
                model_fit = None
                best_aic = float('inf')
                best_model = None
                
                for order, seasonal_order in seasonal_configs:
                    try:
                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore')
                            model = SARIMAX(y_train_sarima, order=order, seasonal_order=seasonal_order, 
                                           enforce_stationarity=False, enforce_invertibility=False)
                            fit = model.fit(disp=False, maxiter=500, method='lbfgs')
                            if fit.aic < best_aic:
                                best_aic = fit.aic
                                best_model = fit
                    except:
                        continue
                
                if best_model:
                    y_pred = best_model.forecast(steps=len(y_test_sarima))
                    # Clipping adaptativo
                    if cv_sarima > 0.3:
                        y_min = max(0, np.percentile(y_train_sarima, 5))
                        y_max = np.percentile(y_train_sarima, 95) * 1.3
                        y_pred = np.clip(y_pred, y_min, y_max)
                    resultados[modelo_nome] = {
                        'model': best_model,
                        'predictions': y_pred,
                        'y_test': y_test_sarima,
                        'metrics': calculate_metrics(y_test_sarima, y_pred, y_train_original)
                    }
            
            elif modelo_nome == 'HoltWinters':
                from statsmodels.tsa.holtwinters import ExponentialSmoothing
                y_train_hw = df_train['Demanda'].values
                y_test_hw = df_test['Demanda'].values
                
                cv_hw = np.std(y_train_hw) / np.mean(y_train_hw) if np.mean(y_train_hw) > 0 else 0
                
                # Configuração adaptativa baseada em volatilidade
                model_fit = None
                if cv_hw > 0.5:
                    # Alta volatilidade: modelo simples sem sazonalidade
                    try:
                        model = ExponentialSmoothing(y_train_hw, trend=None, seasonal=None)
                        model_fit = model.fit(smoothing_level=0.3)  # Suavização forte
                    except:
                        pass
                elif cv_hw > 0.3:
                    # Volatilidade média: trend mas sem sazonalidade
                    try:
                        model = ExponentialSmoothing(y_train_hw, trend='add', seasonal=None, damped_trend=True)
                        model_fit = model.fit()
                    except:
                        try:
                            model = ExponentialSmoothing(y_train_hw, trend=None, seasonal=None)
                            model_fit = model.fit()
                        except:
                            pass
                else:
                    # Baixa volatilidade: modelo completo
                    try:
                        if len(y_train_hw) >= 14:
                            model = ExponentialSmoothing(y_train_hw, seasonal_periods=7, trend='add', seasonal='add', damped_trend=True)
                        else:
                            model = ExponentialSmoothing(y_train_hw, trend='add', seasonal=None)
                        model_fit = model.fit()
                    except:
                        try:
                            model = ExponentialSmoothing(y_train_hw, trend=None, seasonal=None)
                            model_fit = model.fit()
                        except:
                            pass
                
                if model_fit:
                    y_pred = model_fit.forecast(steps=len(y_test_hw))
                    # Clipping
                    if cv_hw > 0.3:
                        y_min = max(0, np.percentile(y_train_hw, 5))
                        y_max = np.percentile(y_train_hw, 95) * 1.3
                        y_pred = np.clip(y_pred, y_min, y_max)
                    resultados[modelo_nome] = {
                        'model': model_fit,
                        'predictions': y_pred,
                        'y_test': y_test_hw,
                        'metrics': calculate_metrics(y_test_hw, y_pred, y_train_original)
                    }
        
        except Exception as e:
            st.warning(f"⚠️ Erro ao treinar {modelo_nome} para {turno_nome}: {str(e)}")
    
    return resultados, None

# ========== BOTÃO DE TREINAMENTO ==========
st.markdown("---")

if st.button("🚀 Iniciar Treinamento", type="primary"):
    
    if len(df) < 30:
        st.error("❌ Dados insuficientes. Mínimo de 30 registros necessários.")
        st.stop()
    
    # Modelos selecionados
    modelos_selecionados = []
    if use_xgboost: modelos_selecionados.append('XGBoost')
    if use_lightgbm: modelos_selecionados.append('LightGBM')
    if use_catboost: modelos_selecionados.append('CatBoost')
    if use_rf: modelos_selecionados.append('RandomForest')
    if use_gb: modelos_selecionados.append('GradientBoosting')
    if use_mlp: modelos_selecionados.append('MLPRegressor')
    if use_ridge: modelos_selecionados.append('Ridge')
    if use_svr: modelos_selecionados.append('SVR')
    if use_elasticnet: modelos_selecionados.append('ElasticNet')
    if use_knn: modelos_selecionados.append('KNN')
    if use_prophet: modelos_selecionados.append('Prophet')
    if use_arima: modelos_selecionados.append('ARIMA')
    if use_sarima: modelos_selecionados.append('SARIMA')
    if use_hw: modelos_selecionados.append('HoltWinters')
    
    if not modelos_selecionados:
        st.error("❌ Selecione pelo menos um modelo!")
        st.stop()
    
    # ========== TREINAR POR GRUPOS (Local/Sublocal/Hora) ==========
    if modo_agrupamento != "Agregado (todos juntos)" and not (treinar_por_turno and tem_turno):
        st.markdown(f"### 🏢 Treinando modelos: {modo_agrupamento}...")
        
        # Determinar grupos baseado no modo
        if modo_agrupamento == "Por Hora":
            grupos = sorted(df['Hora'].dropna().unique().tolist())
            grupo_tipo = 'hora'
        elif modo_agrupamento == "Por Local":
            grupos = sorted(df['Local'].dropna().unique().tolist())
            grupo_tipo = 'local'
        elif modo_agrupamento == "Por Local e Sublocal":
            df_grupos = df.dropna(subset=['Local', 'Sublocal']).groupby(['Local', 'Sublocal']).size().reset_index()
            grupos = [(row['Local'], row['Sublocal']) for _, row in df_grupos.iterrows()]
            grupo_tipo = 'local_sublocal'
        elif modo_agrupamento == "Por Hora e Local":
            df_grupos = df.dropna(subset=['Hora', 'Local']).groupby(['Hora', 'Local']).size().reset_index()
            grupos = [(int(row['Hora']), row['Local']) for _, row in df_grupos.iterrows()]
            grupo_tipo = 'hora_local'
        elif modo_agrupamento == "Por Hora, Local e Sublocal":
            df_grupos = df.dropna(subset=['Hora', 'Local', 'Sublocal']).groupby(['Hora', 'Local', 'Sublocal']).size().reset_index()
            grupos = [(int(row['Hora']), row['Local'], row['Sublocal']) for _, row in df_grupos.iterrows()]
            grupo_tipo = 'hora_local_sublocal'
        else:
            grupos = []
            grupo_tipo = 'none'
        
        resultados_por_grupo = {}
        metricas_por_grupo = {}
        
        progress_bar = st.progress(0)
        
        for i, grupo in enumerate(grupos):
            # Criar nome e filtrar dados
            if grupo_tipo == 'hora':
                grupo_nome = f"Hora {grupo:02d}"
                df_grupo = df[df['Hora'] == grupo].copy()
            elif grupo_tipo == 'local':
                grupo_nome = str(grupo)
                df_grupo = df[df['Local'] == grupo].copy()
            elif grupo_tipo == 'local_sublocal':
                grupo_nome = f"{grupo[0]} - {grupo[1]}"
                df_grupo = df[(df['Local'] == grupo[0]) & (df['Sublocal'] == grupo[1])].copy()
            elif grupo_tipo == 'hora_local':
                grupo_nome = f"H{grupo[0]:02d} | {grupo[1]}"
                df_grupo = df[(df['Hora'] == grupo[0]) & (df['Local'] == grupo[1])].copy()
            elif grupo_tipo == 'hora_local_sublocal':
                grupo_nome = f"H{grupo[0]:02d} | {grupo[1]} - {grupo[2]}"
                df_grupo = df[(df['Hora'] == grupo[0]) & (df['Local'] == grupo[1]) & (df['Sublocal'] == grupo[2])].copy()
            else:
                continue
            
            if len(df_grupo) < 5:
                progress_bar.progress((i + 1) / len(grupos))
                continue
            
            resultados, erro = treinar_modelos(df_grupo, grupo_nome, modelos_selecionados, train_size, validacao_walk_forward)
            
            if erro:
                pass  # Silencioso para muitos grupos
            elif resultados:
                resultados_por_grupo[grupo_nome] = resultados
                
                # Calcular métricas
                metrics_list = []
                for nome, res in resultados.items():
                    m = res['metrics']
                    # Calcular score combinado: 1/MAPE + 1/SMAPE + 1/MASE
                    mape_val = max(m['mape'], 0.001)
                    smape_val = max(m.get('smape', 0.1), 0.001)
                    mase_val = max(m.get('mase', 1), 0.001)
                    score_comb = (1/mape_val + 1/smape_val + 1/mase_val)
                    
                    metrics_list.append({
                        'Modelo': nome,
                        'MAE': m['mae'],
                        'RMSE': m['rmse'],
                        'MAPE (%)': m['mape'] * 100,
                        'SMAPE (%)': m.get('smape', 0) * 100,
                        'MASE': m.get('mase', 0),
                        'R²': m['r2'],
                        'Score': score_comb
                    })
                
                df_metrics = pd.DataFrame(metrics_list).sort_values('Score', ascending=False)
                metricas_por_grupo[grupo_nome] = df_metrics
            
            progress_bar.progress((i + 1) / len(grupos))
        
        if not resultados_por_grupo:
            st.error("❌ Nenhum grupo foi treinado com sucesso.")
            st.stop()
        
        # Salvar resultados
        st.session_state.resultados_por_grupo = resultados_por_grupo
        st.session_state.metricas_por_grupo = metricas_por_grupo
        st.session_state.modo_previsao = 'grupo'
        st.session_state.modo_agrupamento = modo_agrupamento
        st.session_state.grupo_tipo = grupo_tipo
        st.session_state.modelos_treinados = True
        st.session_state.grupos_treinados = list(resultados_por_grupo.keys())
        st.session_state.horizonte = horizonte
        st.session_state.df_filtrado = df
        
        st.success(f"✅ {len(resultados_por_grupo)} grupos treinados com sucesso!")
        
        # Mostrar resumo
        st.markdown("---")
        st.markdown("### 📊 Resumo por Grupo")
        
        # Criar tabela resumo
        resumo_data = []
        for grupo, df_m in metricas_por_grupo.items():
            best = df_m.iloc[0]
            resumo_data.append({
                'Grupo': grupo,
                'Melhor Modelo': best['Modelo'],
                'MAE': f"{best['MAE']:.2f}",
                'MAPE (%)': f"{best['MAPE (%)']:.1f}"
            })
        
        df_resumo = pd.DataFrame(resumo_data)
        st.dataframe(df_resumo, width='stretch', hide_index=True)
        
        st.info("👉 Vá para **Previsões** para gerar previsões por grupo")
    
    # ========== TREINAR POR TURNO ==========
    elif treinar_por_turno and tem_turno:
        st.markdown("### 🕐 Treinando modelos por turno...")
        
        turnos = sorted(df['Turno'].dropna().unique().tolist())
        resultados_por_turno = {}
        metricas_por_turno = {}
        
        progress_bar = st.progress(0)
        
        for i, turno in enumerate(turnos):
            st.markdown(f"**Turno: {turno}**")
            df_turno = df[df['Turno'] == turno].copy()
            
            resultados, erro = treinar_modelos(df_turno, turno, modelos_selecionados, train_size, validacao_walk_forward)
            
            if erro:
                st.warning(erro)
            elif resultados:
                resultados_por_turno[turno] = resultados
                
                # Calcular métricas
                metrics_list = []
                for nome, res in resultados.items():
                    m = res['metrics']
                    mape_val = max(m['mape'], 0.001)
                    smape_val = max(m.get('smape', 0.1), 0.001)
                    mase_val = max(m.get('mase', 1), 0.001)
                    score_comb = (1/mape_val + 1/smape_val + 1/mase_val)
                    
                    metrics_list.append({
                        'Modelo': nome,
                        'MAE': m['mae'],
                        'RMSE': m['rmse'],
                        'MAPE (%)': m['mape'] * 100,
                        'SMAPE (%)': m.get('smape', 0) * 100,
                        'MASE': m.get('mase', 0),
                        'R²': m['r2'],
                        'Score': score_comb
                    })
                
                df_metrics = pd.DataFrame(metrics_list).sort_values('Score', ascending=False)
                metricas_por_turno[turno] = df_metrics
                
                best_model = df_metrics.iloc[0]['Modelo']
                best_mae = df_metrics.iloc[0]['MAE']
                st.success(f"✅ {turno}: Melhor modelo = {best_model} (MAE: {best_mae:.2f})")
            
            progress_bar.progress((i + 1) / len(turnos))
        
        # Salvar resultados
        st.session_state.resultados_por_turno = resultados_por_turno
        st.session_state.metricas_por_turno = metricas_por_turno
        st.session_state.modo_previsao = 'turno'
        st.session_state.modelos_treinados = True
        st.session_state.turnos_treinados = turnos
        st.session_state.horizonte = horizonte
        st.session_state.df_filtrado = df
        
        # Mostrar resumo
        st.markdown("---")
        st.markdown("### 📊 Resumo por Turno")
        
        for turno, df_metrics in metricas_por_turno.items():
            with st.expander(f"🕐 {turno} - {len(df_metrics)} modelos"):
                st.dataframe(df_metrics, width='stretch', hide_index=True)
    
    # ========== TREINAR AGREGADO ==========
    else:
        st.markdown("### 📊 Treinando modelos (agregado)...")
        
        resultados, erro = treinar_modelos(df, 'Agregado', modelos_selecionados, train_size, validacao_walk_forward)
        
        if erro:
            st.error(erro)
            st.stop()
        
        if not resultados:
            st.error("❌ Nenhum modelo foi treinado com sucesso.")
            st.stop()
        
        # Calcular métricas
        metrics_list = []
        for nome, res in resultados.items():
            m = res['metrics']
            mape_val = max(m['mape'], 0.001)
            smape_val = max(m.get('smape', 0.1), 0.001)
            mase_val = max(m.get('mase', 1), 0.001)
            score_comb = (1/mape_val + 1/smape_val + 1/mase_val)
            
            metrics_list.append({
                'Modelo': nome,
                'MAE': m['mae'],
                'RMSE': m['rmse'],
                'MAPE (%)': m['mape'] * 100,
                'SMAPE (%)': m.get('smape', 0) * 100,
                'MASE': m.get('mase', 0),
                'R²': m['r2'],
                'Score': score_comb
            })
        
        df_metrics = pd.DataFrame(metrics_list).sort_values('Score', ascending=False).reset_index(drop=True)
        df_metrics.index = df_metrics.index + 1
        df_metrics.index.name = 'Rank'
        
        best_model = df_metrics.iloc[0]['Modelo']
        
        # ========== CRIAR ENSEMBLE (se habilitado) ==========
        ensemble_info = None
        if usar_ensemble and len(resultados) >= 2:
            st.markdown("---")
            st.markdown("### 🎯 Criando Ensemble para Atingir Metas")
            
            with st.spinner(f"Otimizando ensemble para MAPE < {meta_mape}% e MAE < {meta_mae}..."):
                # Tentar otimizar ensemble para atingir metas
                ensemble_info = otimizar_ensemble_para_meta(
                    resultados, meta_mape, meta_mae, n_modelos_ensemble, metodo_ensemble
                )
            
            if ensemble_info:
                # Adicionar ensemble aos resultados
                resultados['🎯 Ensemble'] = {
                    'modelo': 'Ensemble',
                    'metrics': {
                        'mae': ensemble_info['mae'],
                        'rmse': np.sqrt(np.mean((ensemble_info['y_true'] - ensemble_info['y_pred'])**2)),
                        'mape': ensemble_info['mape'] / 100,  # Converter de % para decimal
                        'smape': ensemble_info.get('smape', 0) / 100,
                        'mase': ensemble_info.get('mase', 1),
                        'r2': 1 - (np.sum((ensemble_info['y_true'] - ensemble_info['y_pred'])**2) / 
                                  np.sum((ensemble_info['y_true'] - np.mean(ensemble_info['y_true']))**2))
                    },
                    'y_pred': ensemble_info['y_pred'],
                    'y_test': ensemble_info['y_true'],
                    'ensemble_modelos': ensemble_info['modelos'],
                    'ensemble_pesos': ensemble_info['pesos'],
                    'ensemble_metodo': metodo_ensemble
                }
                
                # Verificar se atingiu as metas
                meta_mape_ok = ensemble_info['mape'] <= meta_mape
                meta_mae_ok = ensemble_info['mae'] <= meta_mae
                
                # Mostrar resultado do ensemble
                col_e1, col_e2, col_e3 = st.columns(3)
                
                with col_e1:
                    if meta_mape_ok:
                        st.success(f"✅ MAPE: {ensemble_info['mape']:.2f}% (meta: {meta_mape}%)")
                    else:
                        st.warning(f"⚠️ MAPE: {ensemble_info['mape']:.2f}% (meta: {meta_mape}%)")
                
                with col_e2:
                    if meta_mae_ok:
                        st.success(f"✅ MAE: {ensemble_info['mae']:.2f} (meta: {meta_mae})")
                    else:
                        st.warning(f"⚠️ MAE: {ensemble_info['mae']:.2f} (meta: {meta_mae})")
                
                with col_e3:
                    st.info(f"📊 R²: {resultados['🎯 Ensemble']['metrics']['r2']:.4f}")
                
                # Mostrar modelos do ensemble
                st.markdown("**Modelos no Ensemble:**")
                for i, (modelo, peso) in enumerate(zip(ensemble_info['modelos'], ensemble_info['pesos'])):
                    st.write(f"  {i+1}. **{modelo}** - peso: {peso:.3f}")
                
                # Atualizar métricas
                mape_val = max(ensemble_info['mape'] / 100, 0.001)
                smape_val = max(ensemble_info.get('smape', 0.1) / 100, 0.001)
                mase_val = max(ensemble_info.get('mase', 1), 0.001)
                score_comb = (1/mape_val + 1/smape_val + 1/mase_val)
                
                ensemble_metrics = {
                    'Modelo': '🎯 Ensemble',
                    'MAE': ensemble_info['mae'],
                    'RMSE': resultados['🎯 Ensemble']['metrics']['rmse'],
                    'MAPE (%)': ensemble_info['mape'],
                    'SMAPE (%)': ensemble_info.get('smape', 0),
                    'MASE': ensemble_info.get('mase', 0),
                    'R²': resultados['🎯 Ensemble']['metrics']['r2'],
                    'Score': score_comb
                }
                
                # Adicionar ensemble ao DataFrame de métricas
                df_metrics = pd.concat([df_metrics.reset_index(drop=True), pd.DataFrame([ensemble_metrics])], ignore_index=True)
                df_metrics = df_metrics.sort_values('Score', ascending=False).reset_index(drop=True)
                df_metrics.index = df_metrics.index + 1
                df_metrics.index.name = 'Rank'
                
                # Verificar se ensemble é o melhor
                if df_metrics.iloc[0]['Modelo'] == '🎯 Ensemble':
                    best_model = '🎯 Ensemble'
                    st.success("🏆 **Ensemble é o melhor modelo!**")
            else:
                st.warning("⚠️ Não foi possível criar um ensemble válido")
        
        # Salvar
        st.session_state.resultados_modelos = resultados
        st.session_state.df_metrics = df_metrics
        st.session_state.best_model = best_model
        st.session_state.modelos_treinados = True
        st.session_state.modo_previsao = 'agregado'
        st.session_state.horizonte = horizonte
        st.session_state.df_filtrado = df
        st.session_state.ensemble_info = ensemble_info
        
        # Mostrar resultados
        st.markdown("---")
        st.markdown("### 🏆 Ranking Final dos Modelos")
        st.success(f"🥇 **Melhor Modelo: {best_model}** - MAE: {df_metrics.iloc[0]['MAE']:.2f}")
        
        st.dataframe(df_metrics, width='stretch')
        
        # Gráficos
        import plotly.express as px
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_mae = px.bar(df_metrics.reset_index(), x='Modelo', y='MAE', title='MAE (menor = melhor)', color='MAE', color_continuous_scale='RdYlGn_r')
            st.plotly_chart(fig_mae, width='stretch')
        
        with col2:
            fig_mape = px.bar(df_metrics.reset_index(), x='Modelo', y='MAPE (%)', title='MAPE % (menor = melhor)', color='MAPE (%)', color_continuous_scale='RdYlGn_r')
            st.plotly_chart(fig_mape, width='stretch')
    
    st.info("👉 Vá para **Previsões** para gerar previsões")

# ========== MOSTRAR RESULTADOS ANTERIORES ==========
elif 'modelos_treinados' in st.session_state and st.session_state.modelos_treinados:
    st.success("✅ Modelos já treinados")
    
    modo = st.session_state.get('modo_previsao', 'agregado')
    
    if modo == 'turno' and 'metricas_por_turno' in st.session_state:
        st.markdown("### 🕐 Resultados por Turno")
        for turno, df_metrics in st.session_state.metricas_por_turno.items():
            with st.expander(f"🕐 {turno}"):
                st.dataframe(df_metrics, width='stretch', hide_index=True)
    
    elif 'df_metrics' in st.session_state:
        st.markdown("### 🏆 Ranking")
        if 'best_model' in st.session_state:
            st.success(f"🥇 **Melhor: {st.session_state.best_model}**")
        st.dataframe(st.session_state.df_metrics, width='stretch')
    
    if st.button("🔄 Retreinar"):
        st.session_state.modelos_treinados = False
        st.session_state.resultados_modelos = {}
        if 'resultados_por_turno' in st.session_state:
            del st.session_state.resultados_por_turno
        if 'metricas_por_turno' in st.session_state:
            del st.session_state.metricas_por_turno
        st.rerun()
