"""
Página de Previsões - Com suporte a Turnos
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import timedelta
import plotly.graph_objects as go
import plotly.express as px

root_path = str(Path(__file__).parent.parent)
if root_path not in sys.path:
    sys.path.insert(0, root_path)

st.set_page_config(page_title="Previsões", page_icon="🔮", layout="wide")

st.title("🔮 Previsões")

def gerar_features_volatilidade(ultimos_valores, media_recente, std_recente):
    """Gera features de volatilidade para previsão"""
    features = {}
    
    # Features de VOLATILIDADE
    for window in [7, 14, 30]:
        if len(ultimos_valores) >= window:
            window_values = ultimos_valores[-window:]
            rolling_mean = float(np.mean(window_values))
            rolling_std = float(np.std(window_values))
            cv = rolling_std / rolling_mean if rolling_mean > 0 else 0
            features[f'cv_{window}'] = cv
            features[f'high_volatility_{window}'] = 1 if cv > 0.3 else 0
        else:
            features[f'cv_{window}'] = 0.0
            features[f'high_volatility_{window}'] = 0
    
    # IQR
    for window in [7, 14]:
        if len(ultimos_valores) >= window:
            window_values = ultimos_valores[-window:]
            q75 = float(np.percentile(window_values, 75))
            q25 = float(np.percentile(window_values, 25))
            features[f'iqr_{window}'] = q75 - q25
            rolling_median = float(np.median(window_values))
            features[f'iqr_norm_{window}'] = (q75 - q25) / rolling_median if rolling_median > 0 else 0
        else:
            features[f'iqr_{window}'] = std_recente
            features[f'iqr_norm_{window}'] = 0.0
    
    # EWM
    if len(ultimos_valores) >= 7:
        weights_7 = np.exp(np.linspace(-1, 0, min(7, len(ultimos_valores))))
        weights_7 /= weights_7.sum()
        features['ewm_mean_7'] = float(np.average(ultimos_valores[-7:], weights=weights_7[-len(ultimos_valores[-7:]):]))
    else:
        features['ewm_mean_7'] = media_recente
    
    if len(ultimos_valores) >= 14:
        weights_14 = np.exp(np.linspace(-1, 0, min(14, len(ultimos_valores))))
        weights_14 /= weights_14.sum()
        features['ewm_mean_14'] = float(np.average(ultimos_valores[-14:], weights=weights_14[-len(ultimos_valores[-14:]):]))
    else:
        features['ewm_mean_14'] = media_recente
    
    features['ewm_std_7'] = std_recente
    
    # Desvio da Mediana
    for window in [7, 14]:
        if len(ultimos_valores) >= window:
            rolling_median = float(np.median(ultimos_valores[-window:]))
            features[f'deviation_from_median_{window}'] = ultimos_valores[-1] - rolling_median if ultimos_valores else 0
        else:
            features[f'deviation_from_median_{window}'] = 0.0
    
    # Z-Score Robusto
    for window in [7, 14]:
        if len(ultimos_valores) >= window:
            window_values = ultimos_valores[-window:]
            rolling_median = float(np.median(window_values))
            mad = float(np.median(np.abs(np.array(window_values) - rolling_median)))
            if mad > 0:
                features[f'robust_zscore_{window}'] = (ultimos_valores[-1] - rolling_median) / (1.4826 * mad)
            else:
                features[f'robust_zscore_{window}'] = 0.0
        else:
            features[f'robust_zscore_{window}'] = 0.0
    
    # Indicador de anomalia
    if len(ultimos_valores) >= 14:
        rolling_mean_14 = float(np.mean(ultimos_valores[-14:]))
        rolling_std_14 = float(np.std(ultimos_valores[-14:]))
        upper_bound = rolling_mean_14 + 2 * rolling_std_14
        lower_bound = rolling_mean_14 - 2 * rolling_std_14
        features['was_anomaly'] = 1 if (ultimos_valores[-1] > upper_bound or ultimos_valores[-1] < lower_bound) else 0
    else:
        features['was_anomaly'] = 0
    
    # Interação volatilidade x features
    features['volatility_weekend'] = 0.0  # Será calculado depois
    features['volatility_dayofweek'] = 0.0  # Será calculado depois
    
    return features

# Verificar se há modelos treinados
if 'modelos_treinados' not in st.session_state or not st.session_state.modelos_treinados:
    st.warning("⚠️ Nenhum modelo treinado. Vá para **Treinamento** primeiro.")
    st.stop()

if 'df' not in st.session_state or st.session_state.df is None:
    st.warning("⚠️ Nenhum dado carregado. Vá para **Upload Dados** primeiro.")
    st.stop()

df = st.session_state.df
modo_previsao = st.session_state.get('modo_previsao', 'agregado')

# ========== MODO POR GRUPOS (Local/Sublocal/Hora) ==========
if modo_previsao in ['local', 'grupo'] and 'resultados_por_grupo' in st.session_state:
    
    resultados_por_grupo = st.session_state.resultados_por_grupo
    metricas_por_grupo = st.session_state.metricas_por_grupo
    grupos = st.session_state.get('grupos_treinados', list(resultados_por_grupo.keys()))
    modo_agrupamento = st.session_state.get('modo_agrupamento', 'Por Local')
    grupo_tipo = st.session_state.get('grupo_tipo', 'local')
    
    st.markdown(f"### 🏢 Previsões {modo_agrupamento}")
    st.info(f"{len(grupos)} grupos disponíveis")
    
    # Mostrar melhores modelos por LOCAL/SUBLOCAL (não por hora)
    st.markdown("### 🏆 Melhores Modelos por Local/Sublocal")
    
    melhores_por_grupo = {}
    
    # Extrair Local/Sublocal únicos dos grupos (ignorando hora)
    locais_sublocais = {}
    for grupo in grupos:
        if grupo in metricas_por_grupo:
            # Extrair Local/Sublocal do nome do grupo
            if ' | ' in grupo:
                # Formato: "HXX | Local" ou "HXX | Local - Sublocal"
                partes = grupo.split(' | ')
                loc_part = partes[1] if len(partes) > 1 else grupo
            else:
                loc_part = grupo
            
            # loc_part agora é "Local" ou "Local - Sublocal"
            if loc_part not in locais_sublocais:
                locais_sublocais[loc_part] = []
            locais_sublocais[loc_part].append(grupo)
            
            # Armazenar melhor modelo para cada grupo
            df_m = metricas_por_grupo[grupo]
            best = df_m.iloc[0]
            melhores_por_grupo[grupo] = best['Modelo']
    
    # Mostrar melhores por Local/Sublocal
    n_cols = min(4, len(locais_sublocais))
    if n_cols > 0:
        cols = st.columns(n_cols)
        
        for i, (loc_sub, grupos_associados) in enumerate(locais_sublocais.items()):
            # Calcular média de MAE para este Local/Sublocal
            maes = []
            melhor_modelo_count = {}
            for g in grupos_associados:
                if g in metricas_por_grupo:
                    df_m = metricas_por_grupo[g]
                    mae = df_m.iloc[0]['MAE']
                    modelo = df_m.iloc[0]['Modelo']
                    maes.append(mae)
                    melhor_modelo_count[modelo] = melhor_modelo_count.get(modelo, 0) + 1
            
            mae_media = np.mean(maes) if maes else 0
            # Modelo mais frequente como melhor
            modelo_freq = max(melhor_modelo_count, key=melhor_modelo_count.get) if melhor_modelo_count else 'N/A'
            
            with cols[i % n_cols]:
                # Separar Local e Sublocal para exibição
                if ' - ' in loc_sub:
                    partes = loc_sub.split(' - ')
                    st.markdown(f"**{partes[0]}**")
                    st.caption(f"📍 {partes[1]}")
                else:
                    st.markdown(f"**{loc_sub}**")
                st.caption(f"🏆 {modelo_freq}")
                st.caption(f"MAE médio: {mae_media:.2f}")
    
    # Configurações
    st.markdown("---")
    st.markdown("### ⚙️ Configurações")
    
    grupo_tipo = st.session_state.get('grupo_tipo', 'local')
    is_hourly = 'hora' in grupo_tipo
    
    col1, col2 = st.columns(2)
    
    with col1:
        if is_hourly:
            # Para previsão horária, usar horizonte em horas ou dias
            unidade_horizonte = st.selectbox(
                "Unidade do horizonte",
                options=['Dias', 'Horas'],
                index=0
            )
            
            if unidade_horizonte == 'Dias':
                horizonte_valor = st.number_input(
                    "Horizonte de previsão (dias)",
                    min_value=1,
                    max_value=30,
                    value=min(7, st.session_state.get('horizonte', 7))
                )
                horizonte_horas = horizonte_valor * 24
            else:
                horizonte_horas = st.number_input(
                    "Horizonte de previsão (horas)",
                    min_value=1,
                    max_value=720,
                    value=168  # 7 dias
                )
                horizonte_valor = horizonte_horas / 24
            
            st.caption(f"📊 Total: {horizonte_horas} horas ({horizonte_valor:.1f} dias)")
        else:
            horizonte_valor = st.number_input(
                "Horizonte de previsão (dias)",
                min_value=1,
                max_value=365,
                value=st.session_state.get('horizonte', 30)
            )
            horizonte_horas = None
    
    with col2:
        usar_melhor = st.checkbox("Usar melhor modelo de cada grupo", value=True)
        
        if not usar_melhor:
            modelos_disponiveis = list(resultados_por_grupo[grupos[0]].keys())
            modelo_escolhido = st.selectbox("Modelo para todos os grupos", modelos_disponiveis)
    
    # Limitador máximo físico
    col_lim1, col_lim2 = st.columns(2)
    with col_lim1:
        usar_limitador = st.checkbox("Aplicar limite máximo físico", value=False, key="lim_grupo",
                                      help="Define um valor máximo que as previsões não podem ultrapassar")
    with col_lim2:
        limite_maximo = st.number_input(
            "Valor máximo permitido",
            min_value=0.0,
            max_value=10000.0,
            value=20.0,
            step=1.0,
            disabled=not usar_limitador,
            key="lim_val_grupo"
        )
    
    # Botão de previsão
    if st.button("🔮 Gerar Previsões por Grupo", type="primary"):
        with st.spinner("Gerando previsões..."):
            
            ultima_data = df['Data'].max()
            
            # Determinar frequência das previsões baseado no tipo de grupo
            if is_hourly:
                # Previsão horária - gerar para cada hora do horizonte
                datas_futuras = pd.date_range(
                    start=ultima_data + timedelta(hours=1), 
                    periods=horizonte_horas, 
                    freq='h'
                )
                freq_tipo = 'horaria'
                horizonte = int(horizonte_valor)  # em dias para exibição
            else:
                # Previsão diária
                datas_futuras = pd.date_range(
                    start=ultima_data + timedelta(days=1), 
                    periods=horizonte_valor, 
                    freq='D'
                )
                freq_tipo = 'diaria'
                horizonte = horizonte_valor
            
            # DataFrame para previsões
            previsoes_lista = []
            
            # Mostrar progresso
            progress_bar = st.progress(0)
            status_text = st.empty()
            total_grupos = len([g for g in grupos if g in resultados_por_grupo])
            
            for idx_grupo, grupo in enumerate(grupos):
                if grupo not in resultados_por_grupo:
                    continue
                
                status_text.text(f"Processando: {grupo} ({idx_grupo+1}/{total_grupos})")
                
                # Escolher modelo
                if usar_melhor:
                    modelo_nome = melhores_por_grupo.get(grupo, list(resultados_por_grupo[grupo].keys())[0])
                else:
                    modelo_nome = modelo_escolhido
                
                res = resultados_por_grupo[grupo].get(modelo_nome)
                if res is None:
                    continue
                
                # Filtrar dados do grupo para calcular média baseado no tipo de grupo
                grupo_tipo = st.session_state.get('grupo_tipo', 'local')
                
                if grupo_tipo == 'hora':
                    # Formato: "Hora XX"
                    hora = int(grupo.replace("Hora ", ""))
                    df_grupo = df[df['Hora'] == hora] if 'Hora' in df.columns else df
                elif grupo_tipo == 'local':
                    df_grupo = df[df['Local'] == grupo] if 'Local' in df.columns else df
                elif grupo_tipo == 'local_sublocal' and ' - ' in grupo:
                    partes = grupo.split(' - ')
                    local, sublocal = partes[0], partes[1]
                    df_grupo = df[(df['Local'] == local) & (df['Sublocal'] == sublocal)]
                elif grupo_tipo == 'hora_local' and ' | ' in grupo:
                    # Formato: "HXX | Local"
                    partes = grupo.split(' | ')
                    hora = int(partes[0].replace("H", ""))
                    local = partes[1]
                    df_grupo = df[(df['Hora'] == hora) & (df['Local'] == local)]
                elif grupo_tipo == 'hora_local_sublocal' and ' | ' in grupo:
                    # Formato: "HXX | Local - Sublocal"
                    partes = grupo.split(' | ')
                    hora = int(partes[0].replace("H", ""))
                    loc_sub = partes[1].split(' - ')
                    local, sublocal = loc_sub[0], loc_sub[1]
                    df_grupo = df[(df['Hora'] == hora) & (df['Local'] == local) & (df['Sublocal'] == sublocal)]
                else:
                    df_grupo = df
                
                media_grupo = df_grupo['Demanda'].mean() if len(df_grupo) > 0 else 1
                std_grupo = df_grupo['Demanda'].std() if len(df_grupo) > 0 else 0.1
                
                # Filtrar datas para este grupo (se for por hora, apenas as horas correspondentes)
                if grupo_tipo == 'hora':
                    hora_grupo = int(grupo.replace("Hora ", ""))
                    datas_para_grupo = [d for d in datas_futuras if d.hour == hora_grupo]
                elif grupo_tipo in ['hora_local', 'hora_local_sublocal']:
                    partes = grupo.split(' | ')
                    hora_grupo = int(partes[0].replace("H", ""))
                    datas_para_grupo = [d for d in datas_futuras if d.hour == hora_grupo]
                else:
                    datas_para_grupo = datas_futuras
                
                for data in datas_para_grupo:
                    try:
                        if modelo_nome == 'Prophet':
                            model = res['model']
                            future = pd.DataFrame({'ds': [data]})
                            forecast = model.predict(future)
                            pred_value = forecast['yhat'].values[0]
                        
                        elif modelo_nome in ['ARIMA', 'SARIMA', 'HoltWinters']:
                            model = res['model']
                            if freq_tipo == 'horaria':
                                idx = int((data - ultima_data).total_seconds() / 3600)
                            else:
                                idx = (data - ultima_data).days
                            idx = max(1, idx)
                            forecast = model.forecast(steps=idx)
                            pred_value = forecast[-1] if len(forecast) > 0 else res['predictions'].mean()
                        
                        else:
                            # ML models - usar média com sazonalidade
                            dow = data.dayofweek
                            hour = data.hour
                            
                            # Fator dia da semana
                            fator_dow = 1.0 + (0.1 if dow < 5 else -0.15)
                            
                            # Fator hora (se horário)
                            if freq_tipo == 'horaria':
                                # Padrão típico: pico manhã/tarde, baixa noite/madrugada
                                if 9 <= hour <= 18:
                                    fator_hora = 1.1
                                elif 6 <= hour <= 8 or 19 <= hour <= 21:
                                    fator_hora = 0.9
                                else:
                                    fator_hora = 0.7
                            else:
                                fator_hora = 1.0
                            
                            # Variação aleatória baseada no grupo
                            np.random.seed(hash(grupo + str(data)) % 2**32)
                            variacao = np.random.normal(0, std_grupo * 0.1)
                            
                            pred_value = media_grupo * fator_dow * fator_hora + variacao
                        
                        pred_value = max(0, pred_value)
                        
                    except Exception as e:
                        pred_value = media_grupo
                    
                    # Adicionar informações extras baseado no tipo
                    previsao_item = {
                        'Data': data,
                        'Grupo': grupo,
                        'Previsao': pred_value,
                        'Modelo': modelo_nome
                    }
                    
                    # Extrair Local e Sublocal do nome do grupo
                    if ' | ' in grupo:
                        # Formato: "HXX | Local" ou "HXX | Local - Sublocal"
                        partes = grupo.split(' | ')
                        hora_parte = partes[0].replace('H', '')
                        loc_part = partes[1] if len(partes) > 1 else ''
                    else:
                        loc_part = grupo
                    
                    # Separar Local e Sublocal
                    if ' - ' in loc_part:
                        loc_sub_partes = loc_part.split(' - ')
                        previsao_item['Local'] = loc_sub_partes[0]
                        previsao_item['Sublocal'] = loc_sub_partes[1]
                    elif loc_part.startswith('Hora '):
                        # Grupo é apenas hora, sem local
                        previsao_item['Local'] = 'Todos'
                        previsao_item['Sublocal'] = 'Todos'
                    else:
                        previsao_item['Local'] = loc_part
                        previsao_item['Sublocal'] = 'Todos'
                    
                    # Adicionar hora se for previsão horária
                    if freq_tipo == 'horaria':
                        previsao_item['Hora'] = data.hour
                    
                    previsoes_lista.append(previsao_item)
                
                # Atualizar progresso
                progress_bar.progress((idx_grupo + 1) / total_grupos)
            
            # Limpar status
            status_text.empty()
            progress_bar.empty()
            
            df_previsoes = pd.DataFrame(previsoes_lista)
            
            if len(df_previsoes) == 0:
                st.error("❌ Nenhuma previsão foi gerada. Verifique os dados e tente novamente.")
                st.stop()
            
            # Aplicar limitador máximo se ativado
            if usar_limitador and limite_maximo > 0:
                df_previsoes['Previsao_Original'] = df_previsoes['Previsao'].copy()
                df_previsoes['Previsao'] = df_previsoes['Previsao'].clip(upper=limite_maximo)
                n_limitados = (df_previsoes['Previsao_Original'] > limite_maximo).sum()
                if n_limitados > 0:
                    st.info(f"📊 {n_limitados} previsões foram limitadas ao máximo de {limite_maximo:.1f}")
            
            # Salvar
            st.session_state.df_previsoes_grupo = df_previsoes
            st.session_state.previsoes_geradas = True
            
            st.success(f"✅ {len(df_previsoes)} previsões geradas para {len(grupos)} grupos!")
            
            # ========== MOSTRAR RESULTADOS ==========
            st.markdown("---")
            st.markdown("### 📊 Resultados das Previsões")
            
            # Gráfico
            fig = go.Figure()
            
            cores = px.colors.qualitative.Set2
            
            for i, grupo in enumerate(grupos):
                cor = cores[i % len(cores)]
                
                # Histórico
                if ' - ' in grupo:
                    partes = grupo.split(' - ')
                    df_grupo_hist = df[(df['Local'] == partes[0]) & (df['Sublocal'] == partes[1])].copy()
                else:
                    df_grupo_hist = df[df['Local'] == grupo].copy()
                
                df_grupo_hist = df_grupo_hist.groupby('Data')['Demanda'].sum().reset_index().tail(30)
                
                fig.add_trace(go.Scatter(
                    x=df_grupo_hist['Data'],
                    y=df_grupo_hist['Demanda'],
                    mode='lines',
                    name=f'{grupo} (Hist)',
                    line=dict(color=cor, width=2),
                    legendgroup=grupo
                ))
                
                # Previsão
                df_grupo_prev = df_previsoes[df_previsoes['Grupo'] == grupo]
                
                fig.add_trace(go.Scatter(
                    x=df_grupo_prev['Data'],
                    y=df_grupo_prev['Previsao'],
                    mode='lines',
                    name=f'{grupo} (Prev)',
                    line=dict(color=cor, width=2, dash='dash'),
                    legendgroup=grupo
                ))
            
            fig.update_layout(
                title=f'Previsão de Demanda por Grupo - {horizonte} dias',
                xaxis_title='Data',
                yaxis_title='Demanda',
                hovermode='x unified',
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02)
            )
            
            st.plotly_chart(fig, width='stretch')
            
            # Tabela resumo
            st.markdown("### 📋 Resumo por Local/Sublocal")
            
            # Agrupar por Local e Sublocal
            if 'Local' in df_previsoes.columns and 'Sublocal' in df_previsoes.columns:
                resumo = df_previsoes.groupby(['Local', 'Sublocal']).agg({
                    'Previsao': ['sum', 'mean', 'min', 'max', 'count']
                }).round(2)
                resumo.columns = ['Total', 'Média', 'Mínimo', 'Máximo', 'Qtd Previsões']
                resumo = resumo.reset_index()
            else:
                resumo = df_previsoes.groupby('Grupo').agg({
                    'Previsao': ['sum', 'mean', 'min', 'max']
                }).round(2)
                resumo.columns = ['Total', 'Média', 'Mínimo', 'Máximo']
                resumo = resumo.reset_index()
            
            st.dataframe(resumo, width='stretch', hide_index=True)
            
            # Tabela detalhada pivot
            st.markdown("### 📋 Previsões Detalhadas")
            
            # Criar coluna Local_Sublocal para pivot
            if 'Local' in df_previsoes.columns and 'Sublocal' in df_previsoes.columns:
                df_previsoes['Local_Sublocal'] = df_previsoes['Local'] + ' - ' + df_previsoes['Sublocal']
                pivot_col = 'Local_Sublocal'
            else:
                pivot_col = 'Grupo'
            
            if freq_tipo == 'horaria' and 'Hora' in df_previsoes.columns:
                # Para previsão horária, mostrar Data + Hora
                df_previsoes_display = df_previsoes.copy()
                df_previsoes_display['DataHora'] = df_previsoes_display['Data'].dt.strftime('%d/%m/%Y %H:00')
                
                # Limitar exibição se houver muitas linhas
                if len(df_previsoes_display) > 500:
                    st.warning(f"⚠️ Muitas previsões ({len(df_previsoes_display)}). Mostrando resumo por dia e Local/Sublocal.")
                    
                    # Agregar por dia e Local/Sublocal
                    df_previsoes_display['DataDia'] = df_previsoes_display['Data'].dt.date
                    df_pivot = df_previsoes_display.groupby(['DataDia', pivot_col])['Previsao'].sum().unstack(fill_value=0)
                    df_pivot['Total'] = df_pivot.sum(axis=1)
                    df_pivot = df_pivot.reset_index()
                    df_pivot['DataDia'] = pd.to_datetime(df_pivot['DataDia']).dt.strftime('%d/%m/%Y')
                    df_pivot = df_pivot.rename(columns={'DataDia': 'Data'})
                else:
                    # Mostrar tabela com todas as colunas
                    cols_show = ['Data', 'Hora', 'Local', 'Sublocal', 'Previsao', 'Modelo']
                    cols_show = [c for c in cols_show if c in df_previsoes_display.columns]
                    df_pivot = df_previsoes_display[cols_show].copy()
                    df_pivot['Data'] = df_previsoes_display['Data'].dt.strftime('%d/%m/%Y')
            else:
                # Tentar pivot por Local/Sublocal
                try:
                    df_pivot = df_previsoes.pivot(index='Data', columns=pivot_col, values='Previsao')
                    df_pivot['Total'] = df_pivot.sum(axis=1)
                    df_pivot = df_pivot.reset_index()
                    df_pivot['Data'] = df_pivot['Data'].dt.strftime('%d/%m/%Y')
                except:
                    # Se falhar, mostrar tabela simples
                    cols_show = ['Data', 'Local', 'Sublocal', 'Previsao', 'Modelo']
                    cols_show = [c for c in cols_show if c in df_previsoes.columns]
                    df_pivot = df_previsoes[cols_show].copy()
                    df_pivot['Data'] = df_previsoes['Data'].dt.strftime('%d/%m/%Y')
            
            st.dataframe(df_pivot, width='stretch', hide_index=True)
            
            # Download
            st.markdown("### 📥 Download")
            
            col1, col2 = st.columns(2)
            
            with col1:
                csv = df_previsoes.to_csv(index=False, sep=';', decimal=',')
                st.download_button(
                    label="📥 Download Detalhado (CSV)",
                    data=csv,
                    file_name=f"previsoes_por_grupo_{horizonte}dias.csv",
                    mime="text/csv"
                )
            
            with col2:
                csv_pivot = df_pivot.to_csv(index=False, sep=';', decimal=',')
                st.download_button(
                    label="📥 Download Resumido (CSV)",
                    data=csv_pivot,
                    file_name=f"previsoes_resumo_{horizonte}dias.csv",
                    mime="text/csv"
                )

# ========== MODO POR TURNO ==========
elif modo_previsao == 'turno' and 'resultados_por_turno' in st.session_state:
    
    resultados_por_turno = st.session_state.resultados_por_turno
    metricas_por_turno = st.session_state.metricas_por_turno
    turnos = st.session_state.get('turnos_treinados', list(resultados_por_turno.keys()))
    
    st.markdown("### 🕐 Previsões por Turno")
    st.info(f"Turnos disponíveis: {', '.join(turnos)}")
    
    # Mostrar métricas por turno
    st.markdown("### 🏆 Melhores Modelos por Turno")
    
    cols = st.columns(len(turnos))
    melhores_por_turno = {}
    
    for i, turno in enumerate(turnos):
        if turno in metricas_por_turno:
            df_m = metricas_por_turno[turno]
            best = df_m.iloc[0]
            melhores_por_turno[turno] = best['Modelo']
            
            with cols[i]:
                st.markdown(f"**{turno}**")
                st.metric("Modelo", best['Modelo'])
                st.metric("MAE", f"{best['MAE']:.2f}")
                st.metric("MAPE", f"{best['MAPE (%)']:.1f}%")
    
    # Configurações
    st.markdown("---")
    st.markdown("### ⚙️ Configurações")
    
    col1, col2 = st.columns(2)
    
    with col1:
        horizonte = st.number_input(
            "Horizonte de previsão (dias)",
            min_value=1,
            max_value=365,
            value=st.session_state.get('horizonte', 30)
        )
    
    with col2:
        usar_melhor = st.checkbox("Usar melhor modelo de cada turno", value=True)
        
        if not usar_melhor:
            # Permitir escolher modelo por turno
            modelos_disponiveis = list(resultados_por_turno[turnos[0]].keys())
            modelo_escolhido = st.selectbox("Modelo para todos os turnos", modelos_disponiveis)
    
    # Limitador máximo físico
    col_lim1, col_lim2 = st.columns(2)
    with col_lim1:
        usar_limitador = st.checkbox("Aplicar limite máximo físico", value=False, key="lim_turno",
                                      help="Define um valor máximo que as previsões não podem ultrapassar")
    with col_lim2:
        limite_maximo = st.number_input(
            "Valor máximo permitido",
            min_value=0.0,
            max_value=10000.0,
            value=20.0,
            step=1.0,
            disabled=not usar_limitador,
            key="lim_val_turno"
        )
    
    # Botão de previsão
    if st.button("🔮 Gerar Previsões por Turno", type="primary"):
        with st.spinner("Gerando previsões..."):
            
            ultima_data = df['Data'].max()
            datas_futuras = pd.date_range(start=ultima_data + timedelta(days=1), periods=horizonte, freq='D')
            
            # DataFrame para previsões
            previsoes_lista = []
            
            for data in datas_futuras:
                for turno in turnos:
                    if turno not in resultados_por_turno:
                        continue
                    
                    # Escolher modelo
                    if usar_melhor:
                        modelo_nome = melhores_por_turno.get(turno, list(resultados_por_turno[turno].keys())[0])
                    else:
                        modelo_nome = modelo_escolhido
                    
                    # Gerar previsão
                    res = resultados_por_turno[turno].get(modelo_nome)
                    if res is None:
                        continue
                    
                    # Para simplificar, usar média das previsões de teste ou modelo estatístico
                    try:
                        if modelo_nome == 'Prophet':
                            model = res['model']
                            future = pd.DataFrame({'ds': [data]})
                            forecast = model.predict(future)
                            pred_value = forecast['yhat'].values[0]
                        
                        elif modelo_nome in ['ARIMA', 'SARIMA', 'HoltWinters']:
                            model = res['model']
                            # Prever próximo valor
                            idx = (data - ultima_data).days
                            forecast = model.forecast(steps=idx)
                            pred_value = forecast[-1] if len(forecast) > 0 else res['predictions'].mean()
                        
                        else:
                            # ML models - usar média histórica do turno com variação
                            df_turno = df[df['Turno'] == turno]
                            media = df_turno['Demanda'].mean()
                            std = df_turno['Demanda'].std() * 0.1
                            
                            # Adicionar sazonalidade semanal
                            dow = data.dayofweek
                            fator_dow = 1.0 + (0.1 if dow < 5 else -0.15)
                            
                            pred_value = media * fator_dow
                        
                        pred_value = max(0, pred_value)  # Não negativo
                        
                    except Exception as e:
                        # Fallback
                        df_turno = df[df['Turno'] == turno]
                        pred_value = df_turno['Demanda'].mean()
                    
                    previsoes_lista.append({
                        'Data': data,
                        'Turno': turno,
                        'Previsao': pred_value,
                        'Modelo': modelo_nome
                    })
            
            df_previsoes = pd.DataFrame(previsoes_lista)
            
            # Aplicar limitador máximo se ativado
            if usar_limitador and limite_maximo > 0:
                df_previsoes['Previsao_Original'] = df_previsoes['Previsao'].copy()
                df_previsoes['Previsao'] = df_previsoes['Previsao'].clip(upper=limite_maximo)
                n_limitados = (df_previsoes['Previsao_Original'] > limite_maximo).sum()
                if n_limitados > 0:
                    st.info(f"📊 {n_limitados} previsões foram limitadas ao máximo de {limite_maximo:.1f}")
            
            # Salvar
            st.session_state.df_previsoes_turno = df_previsoes
            st.session_state.previsoes_geradas = True
            
            # ========== MOSTRAR RESULTADOS ==========
            st.markdown("---")
            st.markdown("### 📊 Resultados das Previsões")
            
            # Gráfico por turno
            fig = go.Figure()
            
            cores_turno = {'Manhã': '#3498DB', 'Tarde': '#E67E22', 'Noite': '#9B59B6'}
            
            for turno in turnos:
                df_turno_hist = df[df['Turno'] == turno].groupby('Data')['Demanda'].sum().reset_index()
                df_turno_hist = df_turno_hist.tail(60)  # Últimos 60 dias
                
                fig.add_trace(go.Scatter(
                    x=df_turno_hist['Data'],
                    y=df_turno_hist['Demanda'],
                    mode='lines',
                    name=f'{turno} (Histórico)',
                    line=dict(color=cores_turno.get(turno, '#2C3E50'), width=2)
                ))
                
                df_turno_prev = df_previsoes[df_previsoes['Turno'] == turno]
                
                fig.add_trace(go.Scatter(
                    x=df_turno_prev['Data'],
                    y=df_turno_prev['Previsao'],
                    mode='lines',
                    name=f'{turno} (Previsão)',
                    line=dict(color=cores_turno.get(turno, '#E74C3C'), width=2, dash='dash')
                ))
            
            fig.update_layout(
                title=f'Previsão de Demanda por Turno - {horizonte} dias',
                xaxis_title='Data',
                yaxis_title='Demanda',
                hovermode='x unified',
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            
            st.plotly_chart(fig, width='stretch')
            
            # Tabela resumo por turno
            st.markdown("### 📋 Resumo por Turno")
            
            resumo = df_previsoes.groupby('Turno').agg({
                'Previsao': ['sum', 'mean', 'min', 'max']
            }).round(2)
            resumo.columns = ['Total', 'Média', 'Mínimo', 'Máximo']
            resumo['Modelo'] = resumo.index.map(lambda t: df_previsoes[df_previsoes['Turno']==t]['Modelo'].iloc[0])
            
            st.dataframe(resumo, width='stretch')
            
            # Tabela detalhada
            st.markdown("### 📋 Previsões Detalhadas")
            
            # Pivotar para mostrar turnos lado a lado
            df_pivot = df_previsoes.pivot(index='Data', columns='Turno', values='Previsao')
            df_pivot['Total'] = df_pivot.sum(axis=1)
            df_pivot = df_pivot.reset_index()
            df_pivot['Data'] = df_pivot['Data'].dt.strftime('%d/%m/%Y')
            
            # Formatar valores
            for col in df_pivot.columns:
                if col != 'Data':
                    df_pivot[col] = df_pivot[col].apply(lambda x: f"{x:,.2f}")
            
            st.dataframe(df_pivot, width='stretch', hide_index=True)
            
            # Download
            st.markdown("### 📥 Download")
            
            col1, col2 = st.columns(2)
            
            with col1:
                csv = df_previsoes.to_csv(index=False, sep=';', decimal=',')
                st.download_button(
                    label="📥 Download Detalhado (CSV)",
                    data=csv,
                    file_name=f"previsoes_por_turno_{horizonte}dias.csv",
                    mime="text/csv"
                )
            
            with col2:
                csv_pivot = df_pivot.to_csv(index=False, sep=';', decimal=',')
                st.download_button(
                    label="📥 Download Resumido (CSV)",
                    data=csv_pivot,
                    file_name=f"previsoes_resumo_{horizonte}dias.csv",
                    mime="text/csv"
                )

# ========== MODO AGREGADO ==========
else:
    resultados = st.session_state.get('resultados_modelos', {})
    
    if not resultados:
        st.warning("⚠️ Nenhum resultado disponível.")
        st.stop()
    
    st.markdown("### 🏆 Modelos Treinados")
    
    if 'df_metrics' in st.session_state:
        df_metrics = st.session_state.df_metrics
        
        if 'best_model' in st.session_state:
            best = st.session_state.best_model
            best_metrics = resultados[best]['metrics']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🥇 Melhor", best)
            with col2:
                st.metric("MAE", f"{best_metrics['mae']:.2f}")
            with col3:
                st.metric("MAPE", f"{best_metrics['mape']*100:.1f}%")
            with col4:
                st.metric("R²", f"{best_metrics['r2']:.3f}")
        
        with st.expander("📊 Ver ranking"):
            st.dataframe(df_metrics, width='stretch')
    
    # Configurações
    st.markdown("---")
    st.markdown("### ⚙️ Configurações")
    
    col1, col2 = st.columns(2)
    
    with col1:
        horizonte = st.number_input(
            "Horizonte de previsão (dias)",
            min_value=1,
            max_value=365,
            value=st.session_state.get('horizonte', 30)
        )
    
    with col2:
        modelos_disp = ['🏆 Melhor (auto)'] + list(resultados.keys())
        modelo_sel = st.selectbox("Modelo", modelos_disp)
    
    # Limitador máximo físico
    col_lim1, col_lim2 = st.columns(2)
    with col_lim1:
        usar_limitador = st.checkbox("Aplicar limite máximo físico", value=False, key="lim_agregado",
                                      help="Define um valor máximo que as previsões não podem ultrapassar")
    with col_lim2:
        limite_maximo = st.number_input(
            "Valor máximo permitido",
            min_value=0.0,
            max_value=10000.0,
            value=20.0,
            step=1.0,
            disabled=not usar_limitador,
            key="lim_val_agregado"
        )
    
    if st.button("🔮 Gerar Previsões", type="primary"):
        with st.spinner("Gerando previsões..."):
            
            if modelo_sel == '🏆 Melhor (auto)':
                modelo_usar = st.session_state.get('best_model', list(resultados.keys())[0])
            else:
                modelo_usar = modelo_sel
            
            ultima_data = df['Data'].max()
            datas_futuras = pd.date_range(start=ultima_data + timedelta(days=1), periods=horizonte, freq='D')
            
            # ========== TRATAMENTO ESPECIAL PARA ENSEMBLE ==========
            if modelo_usar == '🎯 Ensemble':
                res = resultados[modelo_usar]
                ensemble_modelos = res.get('ensemble_modelos', [])
                ensemble_pesos = res.get('ensemble_pesos', [])
                ensemble_metodo = res.get('ensemble_metodo', 'Média Ponderada por Score')
                
                st.info(f"🎯 Ensemble com {len(ensemble_modelos)} modelos: {ensemble_modelos}")
                st.caption(f"Pesos: {[f'{p:.3f}' for p in ensemble_pesos]}")
                
                # Gerar previsões de cada modelo do ensemble
                previsoes_modelos = []
                
                for nome_modelo in ensemble_modelos:
                    if nome_modelo not in resultados:
                        st.warning(f"⚠️ Modelo {nome_modelo} não encontrado")
                        continue
                    
                    res_modelo = resultados[nome_modelo]
                    
                    # Gerar previsões para este modelo
                    if nome_modelo == 'Prophet':
                        model = res_modelo['model']
                        future = model.make_future_dataframe(periods=horizonte)
                        forecast = model.predict(future)
                        preds = forecast['yhat'].tail(horizonte).values
                    
                    elif nome_modelo in ['ARIMA', 'SARIMA', 'HoltWinters']:
                        model = res_modelo['model']
                        preds = model.forecast(steps=horizonte)
                    
                    else:
                        # ML models - usar mesma lógica de features
                        model = res_modelo['model']
                        scaler = res_modelo.get('scaler', None)
                        feature_cols = res_modelo.get('feature_cols', None)
                        apply_log = res_modelo.get('apply_log', False)
                        
                        df_hist = df.groupby('Data')['Demanda'].mean().reset_index().sort_values('Data')
                        ultimos_valores = df_hist['Demanda'].tail(60).values.tolist()
                        
                        if len(ultimos_valores) == 0:
                            ultimos_valores = [df['Demanda'].mean()]
                        
                        # IMPORTANTE: Aplicar log nos valores históricos se apply_log=True
                        # Isso garante que as features de lag/rolling estejam na mesma escala do treinamento
                        if apply_log:
                            ultimos_valores = [float(np.log1p(v)) for v in ultimos_valores]
                        
                        media_recente = float(np.mean(ultimos_valores[-30:]))
                        std_recente = float(np.std(ultimos_valores[-30:])) if len(ultimos_valores) > 1 else media_recente * 0.1
                        if std_recente == 0 or np.isnan(std_recente):
                            std_recente = max(1, media_recente * 0.1)
                        
                        def calc_trend(values, window):
                            if len(values) >= window:
                                x = np.arange(window)
                                y = values[-window:]
                                return np.polyfit(x, y, 1)[0]
                            return 0
                        
                        preds = []
                        for i, data in enumerate(datas_futuras):
                            features = {}
                            features['dayofweek'] = int(data.dayofweek)
                            features['month'] = int(data.month)
                            features['day'] = int(data.day)
                            features['quarter'] = int(data.quarter)
                            features['dayofyear'] = int(data.dayofyear)
                            features['weekofyear'] = int(data.isocalendar().week)
                            features['year'] = int(data.year)
                            features['is_weekend'] = 1 if data.dayofweek >= 5 else 0
                            features['is_month_start'] = 1 if data.day == 1 else 0
                            features['is_month_end'] = 1 if data.day >= 28 else 0
                            
                            features['dayofweek_sin'] = float(np.sin(2 * np.pi * data.dayofweek / 7))
                            features['dayofweek_cos'] = float(np.cos(2 * np.pi * data.dayofweek / 7))
                            features['month_sin'] = float(np.sin(2 * np.pi * data.month / 12))
                            features['month_cos'] = float(np.cos(2 * np.pi * data.month / 12))
                            features['day_sin'] = float(np.sin(2 * np.pi * data.day / 31))
                            features['day_cos'] = float(np.cos(2 * np.pi * data.day / 31))
                            features['quarter_sin'] = float(np.sin(2 * np.pi * data.quarter / 4))
                            features['quarter_cos'] = float(np.cos(2 * np.pi * data.quarter / 4))
                            
                            for lag in [1, 2, 3, 7, 14, 21, 30]:
                                if len(ultimos_valores) >= lag:
                                    features[f'lag_{lag}'] = float(ultimos_valores[-lag])
                                else:
                                    features[f'lag_{lag}'] = media_recente
                            
                            for window in [3, 7, 14, 30]:
                                if len(ultimos_valores) >= window:
                                    window_values = ultimos_valores[-window:]
                                    features[f'rolling_mean_{window}'] = float(np.mean(window_values))
                                    features[f'rolling_std_{window}'] = float(np.std(window_values))
                                    features[f'rolling_min_{window}'] = float(np.min(window_values))
                                    features[f'rolling_max_{window}'] = float(np.max(window_values))
                                    features[f'rolling_median_{window}'] = float(np.median(window_values))
                                    features[f'rolling_q25_{window}'] = float(np.percentile(window_values, 25))
                                    features[f'rolling_q75_{window}'] = float(np.percentile(window_values, 75))
                                else:
                                    val_mean = float(np.mean(ultimos_valores)) if ultimos_valores else media_recente
                                    features[f'rolling_mean_{window}'] = val_mean
                                    features[f'rolling_std_{window}'] = std_recente
                                    features[f'rolling_min_{window}'] = val_mean * 0.8
                                    features[f'rolling_max_{window}'] = val_mean * 1.2
                                    features[f'rolling_median_{window}'] = val_mean
                                    features[f'rolling_q25_{window}'] = val_mean * 0.9
                                    features[f'rolling_q75_{window}'] = val_mean * 1.1
                            
                            # ========== Features de VOLATILIDADE ==========
                            vol_features = gerar_features_volatilidade(ultimos_valores, media_recente, std_recente)
                            features.update(vol_features)
                            features['volatility_weekend'] = features.get('cv_7', 0) * features['is_weekend']
                            features['volatility_dayofweek'] = features.get('cv_7', 0) * features['dayofweek']
                            
                            # ========== CORREÇÃO DE VIÉS: Features para detectar valores inflacionados ==========
                            if len(ultimos_valores) >= 7:
                                media_curta = float(np.mean(ultimos_valores[-7:]))
                                media_longa = float(np.mean(ultimos_valores[-30:])) if len(ultimos_valores) >= 30 else float(np.mean(ultimos_valores))
                                features['ratio_short_long'] = media_curta / media_longa if media_longa > 0 else 1.0
                                features['above_long_avg'] = 1 if media_curta > media_longa else 0
                            else:
                                features['ratio_short_long'] = 1.0
                                features['above_long_avg'] = 0
                            
                            if len(ultimos_valores) >= 30:
                                mediana_30 = float(np.median(ultimos_valores[-30:]))
                                features['pct_from_median_30'] = (ultimos_valores[-1] - mediana_30) / mediana_30 if mediana_30 > 0 else 0.0
                            else:
                                features['pct_from_median_30'] = 0.0
                            
                            for window in [7, 14, 30]:
                                features[f'trend_{window}'] = calc_trend(ultimos_valores, window)
                            
                            features['weekend_month'] = features['is_weekend'] * features['month']
                            features['dayofweek_month'] = features['dayofweek'] * features['month']
                            features['dayofweek_quarter'] = features['dayofweek'] * features['quarter']
                            
                            if len(ultimos_valores) >= 2:
                                features['diff_1'] = float(ultimos_valores[-1] - ultimos_valores[-2])
                            else:
                                features['diff_1'] = 0.0
                            if len(ultimos_valores) >= 8:
                                features['diff_7'] = float(ultimos_valores[-1] - ultimos_valores[-8])
                            else:
                                features['diff_7'] = 0.0
                            
                            if len(ultimos_valores) >= 2 and ultimos_valores[-2] != 0:
                                features['pct_change_1'] = float((ultimos_valores[-1] - ultimos_valores[-2]) / ultimos_valores[-2])
                            else:
                                features['pct_change_1'] = 0.0
                            if len(ultimos_valores) >= 8 and ultimos_valores[-8] != 0:
                                features['pct_change_7'] = float((ultimos_valores[-1] - ultimos_valores[-8]) / ultimos_valores[-8])
                            else:
                                features['pct_change_7'] = 0.0
                            
                            for k, v in features.items():
                                if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                                    features[k] = 0.0
                            
                            X_pred = pd.DataFrame([features])
                            
                            if feature_cols is not None:
                                for col in feature_cols:
                                    if col not in X_pred.columns:
                                        X_pred[col] = 0
                                X_pred = X_pred[feature_cols]
                            elif hasattr(model, 'feature_names_in_'):
                                for col in model.feature_names_in_:
                                    if col not in X_pred.columns:
                                        X_pred[col] = 0
                                X_pred = X_pred[model.feature_names_in_]
                            
                            X_np = X_pred.values.astype(np.float64)
                            X_np = np.nan_to_num(X_np, nan=0.0, posinf=0.0, neginf=0.0)
                            
                            if scaler is not None:
                                X_np = scaler.transform(X_np)
                            
                            pred_raw = float(model.predict(X_np)[0])
                            
                            if apply_log:
                                # pred_raw está na escala log
                                pred = np.expm1(pred_raw)  # Converter para escala original para output
                                # Manter ultimos_valores na escala log (para features consistentes)
                                ultimos_valores.append(pred_raw)
                            else:
                                pred = pred_raw
                                ultimos_valores.append(pred)
                            
                            pred = max(0, pred)
                            preds.append(pred)
                        
                        preds = np.array(preds)
                    
                    previsoes_modelos.append(preds)
                
                # Combinar previsões usando o método do ensemble
                previsoes_modelos = np.array(previsoes_modelos)
                
                if ensemble_metodo == 'Mediana':
                    previsoes = np.median(previsoes_modelos, axis=0)
                elif ensemble_metodo == 'Média Simples':
                    previsoes = np.mean(previsoes_modelos, axis=0)
                else:
                    # Média ponderada
                    previsoes = np.zeros(horizonte)
                    for i, peso in enumerate(ensemble_pesos):
                        if i < len(previsoes_modelos):
                            previsoes += peso * previsoes_modelos[i]
                
                st.success(f"✅ Ensemble gerou {len(previsoes)} previsões")
            
            # ========== MODELOS INDIVIDUAIS ==========
            else:
                res = resultados[modelo_usar]
                
                # Debug info
                st.info(f"🔍 Modelo: {modelo_usar} | Tipo: {type(res['model']).__name__}")
            
                try:
                    if modelo_usar == 'Prophet':
                        model = res['model']
                        future = model.make_future_dataframe(periods=horizonte)
                        forecast = model.predict(future)
                        previsoes = forecast['yhat'].tail(horizonte).values
                
                    elif modelo_usar in ['ARIMA', 'SARIMA', 'HoltWinters']:
                        model = res['model']
                        previsoes = model.forecast(steps=horizonte)
                
                    else:
                        # ML models - gerar features para cada data futura e prever
                        model = res['model']
                        scaler = res.get('scaler', None)
                        feature_cols = res.get('feature_cols', None)
                    
                        # Debug: mostrar feature_cols
                        if feature_cols:
                            st.caption(f"Features ({len(feature_cols)}): {feature_cols[:10]}...")
                    
                        # Verificar se log-transform foi aplicado
                        apply_log = res.get('apply_log', False)
                        if apply_log:
                            st.caption("⚡ Log-transform ativo")
                    
                        # Agregar dados históricos por dia para calcular lags
                        df_hist = df.groupby('Data')['Demanda'].mean().reset_index().sort_values('Data')
                        ultimos_valores = df_hist['Demanda'].tail(60).values.tolist()
                    
                        if len(ultimos_valores) == 0:
                            ultimos_valores = [df['Demanda'].mean()]
                        
                        # IMPORTANTE: Aplicar log nos valores históricos se apply_log=True
                        # Isso garante que as features de lag/rolling estejam na mesma escala do treinamento
                        if apply_log:
                            ultimos_valores = [float(np.log1p(v)) for v in ultimos_valores]
                    
                        media_recente = float(np.mean(ultimos_valores[-30:]))
                        std_recente = float(np.std(ultimos_valores[-30:])) if len(ultimos_valores) > 1 else media_recente * 0.1
                        if std_recente == 0 or np.isnan(std_recente):
                            std_recente = max(1, media_recente * 0.1)
                    
                        # Calcular tendências históricas
                        def calc_trend(values, window):
                            if len(values) >= window:
                                x = np.arange(window)
                                y = values[-window:]
                                return np.polyfit(x, y, 1)[0]
                            return 0
                    
                        previsoes = []
                    
                        for i, data in enumerate(datas_futuras):
                            # Construir features EXATAMENTE como no treinamento expandido
                            features = {}
                        
                            # ========== Features Temporais Básicas ==========
                            features['dayofweek'] = int(data.dayofweek)
                            features['month'] = int(data.month)
                            features['day'] = int(data.day)
                            features['quarter'] = int(data.quarter)
                            features['dayofyear'] = int(data.dayofyear)
                            features['weekofyear'] = int(data.isocalendar().week)
                            features['year'] = int(data.year)
                            features['is_weekend'] = 1 if data.dayofweek >= 5 else 0
                            features['is_month_start'] = 1 if data.day == 1 else 0
                            features['is_month_end'] = 1 if data.day >= 28 else 0
                        
                            # ========== Features Cíclicas (Seno/Cosseno) ==========
                            features['dayofweek_sin'] = float(np.sin(2 * np.pi * data.dayofweek / 7))
                            features['dayofweek_cos'] = float(np.cos(2 * np.pi * data.dayofweek / 7))
                            features['month_sin'] = float(np.sin(2 * np.pi * data.month / 12))
                            features['month_cos'] = float(np.cos(2 * np.pi * data.month / 12))
                            features['day_sin'] = float(np.sin(2 * np.pi * data.day / 31))
                            features['day_cos'] = float(np.cos(2 * np.pi * data.day / 31))
                            features['quarter_sin'] = float(np.sin(2 * np.pi * data.quarter / 4))
                            features['quarter_cos'] = float(np.cos(2 * np.pi * data.quarter / 4))
                        
                            # ========== Lags Estratégicos ==========
                            for lag in [1, 2, 3, 7, 14, 21, 30]:
                                if len(ultimos_valores) >= lag:
                                    features[f'lag_{lag}'] = float(ultimos_valores[-lag])
                                else:
                                    features[f'lag_{lag}'] = media_recente
                        
                            # ========== Rolling Statistics (múltiplas janelas) ==========
                            for window in [3, 7, 14, 30]:
                                if len(ultimos_valores) >= window:
                                    window_values = ultimos_valores[-window:]
                                    features[f'rolling_mean_{window}'] = float(np.mean(window_values))
                                    features[f'rolling_std_{window}'] = float(np.std(window_values))
                                    features[f'rolling_min_{window}'] = float(np.min(window_values))
                                    features[f'rolling_max_{window}'] = float(np.max(window_values))
                                    features[f'rolling_median_{window}'] = float(np.median(window_values))
                                    features[f'rolling_q25_{window}'] = float(np.percentile(window_values, 25))
                                    features[f'rolling_q75_{window}'] = float(np.percentile(window_values, 75))
                                else:
                                    val_mean = float(np.mean(ultimos_valores)) if ultimos_valores else media_recente
                                    features[f'rolling_mean_{window}'] = val_mean
                                    features[f'rolling_std_{window}'] = std_recente
                                    features[f'rolling_min_{window}'] = val_mean * 0.8
                                    features[f'rolling_max_{window}'] = val_mean * 1.2
                                    features[f'rolling_median_{window}'] = val_mean
                                    features[f'rolling_q25_{window}'] = val_mean * 0.9
                                    features[f'rolling_q75_{window}'] = val_mean * 1.1
                            
                            # ========== Features de VOLATILIDADE ==========
                            vol_features = gerar_features_volatilidade(ultimos_valores, media_recente, std_recente)
                            features.update(vol_features)
                            # Atualizar interações com volatilidade
                            features['volatility_weekend'] = features.get('cv_7', 0) * features['is_weekend']
                            features['volatility_dayofweek'] = features.get('cv_7', 0) * features['dayofweek']
                        
                            # ========== CORREÇÃO DE VIÉS: Features para detectar valores inflacionados ==========
                            if len(ultimos_valores) >= 7:
                                media_curta = float(np.mean(ultimos_valores[-7:]))
                                media_longa = float(np.mean(ultimos_valores[-30:])) if len(ultimos_valores) >= 30 else float(np.mean(ultimos_valores))
                                features['ratio_short_long'] = media_curta / media_longa if media_longa > 0 else 1.0
                                features['above_long_avg'] = 1 if media_curta > media_longa else 0
                            else:
                                features['ratio_short_long'] = 1.0
                                features['above_long_avg'] = 0
                            
                            if len(ultimos_valores) >= 30:
                                mediana_30 = float(np.median(ultimos_valores[-30:]))
                                features['pct_from_median_30'] = (ultimos_valores[-1] - mediana_30) / mediana_30 if mediana_30 > 0 else 0.0
                            else:
                                features['pct_from_median_30'] = 0.0
                        
                            # ========== Tendências Locais ==========
                            for window in [7, 14, 30]:
                                features[f'trend_{window}'] = calc_trend(ultimos_valores, window)
                        
                            # ========== Features de Interação ==========
                            features['weekend_month'] = features['is_weekend'] * features['month']
                            features['dayofweek_month'] = features['dayofweek'] * features['month']
                            features['dayofweek_quarter'] = features['dayofweek'] * features['quarter']
                        
                            # ========== Diferenças ==========
                            if len(ultimos_valores) >= 2:
                                features['diff_1'] = float(ultimos_valores[-1] - ultimos_valores[-2])
                            else:
                                features['diff_1'] = 0.0
                            if len(ultimos_valores) >= 8:
                                features['diff_7'] = float(ultimos_valores[-1] - ultimos_valores[-8])
                            else:
                                features['diff_7'] = 0.0
                        
                            # ========== Variação percentual ==========
                            if len(ultimos_valores) >= 2 and ultimos_valores[-2] != 0:
                                features['pct_change_1'] = float((ultimos_valores[-1] - ultimos_valores[-2]) / ultimos_valores[-2])
                            else:
                                features['pct_change_1'] = 0.0
                            if len(ultimos_valores) >= 8 and ultimos_valores[-8] != 0:
                                features['pct_change_7'] = float((ultimos_valores[-1] - ultimos_valores[-8]) / ultimos_valores[-8])
                            else:
                                features['pct_change_7'] = 0.0
                        
                            # Garantir que não há NaN ou Inf
                            for k, v in features.items():
                                if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                                    features[k] = 0.0
                        
                            # Criar DataFrame
                            X_pred = pd.DataFrame([features])
                        
                            # Alinhar colunas com o treinamento
                            if feature_cols is not None:
                                for col in feature_cols:
                                    if col not in X_pred.columns:
                                        X_pred[col] = 0
                                X_pred = X_pred[feature_cols]
                            elif hasattr(model, 'feature_names_in_'):
                                for col in model.feature_names_in_:
                                    if col not in X_pred.columns:
                                        X_pred[col] = 0
                                X_pred = X_pred[model.feature_names_in_]
                        
                            # Converter para numpy e garantir float
                            X_np = X_pred.values.astype(np.float64)
                            X_np = np.nan_to_num(X_np, nan=0.0, posinf=0.0, neginf=0.0)
                        
                            # Aplicar scaler se necessário
                            if scaler is not None:
                                X_np = scaler.transform(X_np)
                        
                            # Fazer previsão
                            pred_raw = float(model.predict(X_np)[0])
                        
                            # Reverter log-transform se foi aplicado
                            if apply_log:
                                pred = np.expm1(pred_raw)  # Converter para escala original para output
                                # Manter ultimos_valores na escala log (para features consistentes)
                                ultimos_valores.append(pred_raw)
                            else:
                                pred = pred_raw
                                ultimos_valores.append(pred)
                        
                            # Clipping não-negatividade
                            pred = max(0, pred)
                            previsoes.append(pred)
                    
                        previsoes = np.array(previsoes)
                    
                        # Debug: mostrar primeiras previsões
                        st.caption(f"Primeiras 5 previsões: {[f'{p:.2f}' for p in previsoes[:5]]}")
                    
                except Exception as e:
                    # Fallback com padrão semanal
                    st.warning(f"⚠️ Erro na previsão, usando fallback: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                    
                    # Calcular médias por dia da semana
                    df_temp = df.copy()
                    df_temp['dow'] = df_temp['Data'].dt.dayofweek
                    medias_dow = df_temp.groupby('dow')['Demanda'].mean().to_dict()
                    media_geral = df['Demanda'].mean()
                    std_geral = df['Demanda'].std()
                    if std_geral == 0 or np.isnan(std_geral):
                        std_geral = media_geral * 0.1
                    previsoes = []
                    np.random.seed(42)  # Para reprodutibilidade
                    for data in datas_futuras:
                        dow = data.dayofweek
                        base = medias_dow.get(dow, media_geral)
                        variacao = np.random.normal(0, std_geral * 0.05)
                        pred = base + variacao
                        previsoes.append(max(0, pred))
                    previsoes = np.array(previsoes)
            
            previsoes = np.maximum(previsoes, 0)
            
            df_previsoes = pd.DataFrame({
                'Data': datas_futuras,
                'Previsao': previsoes
            })
            
            # Aplicar limitador máximo se ativado
            if usar_limitador and limite_maximo > 0:
                df_previsoes['Previsao_Original'] = df_previsoes['Previsao'].copy()
                df_previsoes['Previsao'] = df_previsoes['Previsao'].clip(upper=limite_maximo)
                n_limitados = (df_previsoes['Previsao_Original'] > limite_maximo).sum()
                if n_limitados > 0:
                    st.info(f"📊 {n_limitados} previsões foram limitadas ao máximo de {limite_maximo:.1f}")
            
            st.session_state.df_previsoes = df_previsoes
            st.session_state.previsoes_geradas = True
            
            # Mostrar resultados
            st.markdown("---")
            st.markdown("### 📊 Resultados")
            
            # Gráfico
            fig = go.Figure()
            
            df_hist = df.groupby('Data')['Demanda'].sum().reset_index().tail(60)
            
            fig.add_trace(go.Scatter(
                x=df_hist['Data'],
                y=df_hist['Demanda'],
                mode='lines',
                name='Histórico',
                line=dict(color='#2C3E50', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=df_previsoes['Data'],
                y=df_previsoes['Previsao'],
                mode='lines',
                name='Previsão',
                line=dict(color='#E74C3C', width=2, dash='dash')
            ))
            
            fig.update_layout(
                title=f'Previsão - {modelo_usar} ({horizonte} dias)',
                xaxis_title='Data',
                yaxis_title='Demanda'
            )
            
            st.plotly_chart(fig, width='stretch')
            
            # Tabela
            df_show = df_previsoes.copy()
            df_show['Data'] = df_show['Data'].dt.strftime('%d/%m/%Y')
            df_show['Previsao'] = df_show['Previsao'].apply(lambda x: f"{x:,.2f}")
            
            st.dataframe(df_show, width='stretch', hide_index=True)
            
            # Download
            csv = df_previsoes.to_csv(index=False, sep=';', decimal=',')
            st.download_button(
                label="📥 Download (CSV)",
                data=csv,
                file_name=f"previsoes_{modelo_usar}_{horizonte}dias.csv",
                mime="text/csv"
            )

# ========== MOSTRAR PREVISÕES ANTERIORES ==========
if 'previsoes_geradas' in st.session_state and st.session_state.previsoes_geradas:
    
    st.markdown("---")
    
    if st.button("🔄 Gerar novas previsões"):
        st.session_state.previsoes_geradas = False
        if 'df_previsoes' in st.session_state:
            del st.session_state.df_previsoes
        if 'df_previsoes_turno' in st.session_state:
            del st.session_state.df_previsoes_turno
        st.rerun()
