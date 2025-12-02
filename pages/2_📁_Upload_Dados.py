"""
Página de Upload de Dados
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import List, Tuple
from datetime import time

# Adicionar diretório raiz ao path
root_path = str(Path(__file__).parent.parent)
if root_path not in sys.path:
    sys.path.insert(0, root_path)

st.set_page_config(page_title="Upload Dados", page_icon="📁", layout="wide")


# ====== FUNÇÕES DE AGREGAÇÃO ======

def detect_datetime_has_time(df: pd.DataFrame, date_col: str) -> bool:
    """Detecta se a coluna de data contém componente de hora."""
    try:
        sample = df[date_col].head(100)
        if pd.api.types.is_datetime64_any_dtype(sample):
            times = sample.dt.time
            non_midnight = times[times != time(0, 0, 0)]
            return len(non_midnight) > len(sample) * 0.1
        return False
    except:
        return False


def aggregate_to_daily(df: pd.DataFrame, date_col: str, value_col: str, 
                       method: str = 'sum', group_cols: list = None) -> pd.DataFrame:
    """Agrega dados para granularidade diária, mantendo agrupamentos."""
    df = df.copy()
    df['Data_Dia'] = pd.to_datetime(df[date_col]).dt.date
    
    agg_funcs = {
        'sum': 'sum',
        'mean': 'mean',
        'max': 'max',
        'min': 'min',
        'p80': lambda x: np.percentile(x, 80)
    }
    
    agg_func = agg_funcs.get(method, 'sum')
    
    # Definir colunas de agrupamento
    group_by = ['Data_Dia']
    if group_cols:
        group_by.extend(group_cols)
    
    df_agg = df.groupby(group_by).agg({value_col: agg_func}).reset_index()
    
    # Renomear coluna de data
    df_agg = df_agg.rename(columns={'Data_Dia': 'Data'})
    df_agg['Data'] = pd.to_datetime(df_agg['Data'])
    
    return df_agg


def _assign_shift(hour: int, turnos: dict) -> str:
    """Atribui turno baseado na hora."""
    for turno_nome, config in turnos.items():
        inicio = config['inicio']
        fim = config['fim']
        
        if inicio <= fim:
            if inicio <= hour < fim:
                return turno_nome
        else:  # Turno atravessa meia-noite
            if hour >= inicio or hour < fim:
                return turno_nome
    
    return list(turnos.keys())[0]


def aggregate_to_shifts(df: pd.DataFrame, date_col: str, value_col: str, 
                       turnos: dict, method: str = 'sum', group_cols: list = None) -> pd.DataFrame:
    """Agrega dados por turno, mantendo agrupamentos."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df['Data_Dia'] = df[date_col].dt.date
    df['Hora'] = df[date_col].dt.hour
    df['Turno'] = df['Hora'].apply(lambda h: _assign_shift(h, turnos))
    
    agg_funcs = {
        'sum': 'sum',
        'mean': 'mean',
        'max': 'max',
        'min': 'min',
        'p80': lambda x: np.percentile(x, 80)
    }
    
    agg_func = agg_funcs.get(method, 'sum')
    
    # Definir colunas de agrupamento
    group_by = ['Data_Dia', 'Turno']
    if group_cols:
        group_by.extend(group_cols)
    
    df_agg = df.groupby(group_by).agg({value_col: agg_func}).reset_index()
    
    # Renomear coluna de data
    df_agg = df_agg.rename(columns={'Data_Dia': 'Data'})
    df_agg['Data'] = pd.to_datetime(df_agg['Data'])
    
    return df_agg


def aggregate_hourly(df: pd.DataFrame, date_col: str, value_col: str, 
                    method: str = 'sum', group_cols: list = None) -> pd.DataFrame:
    """Agrega dados por hora, mantendo agrupamentos."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df['Data_Hora'] = df[date_col].dt.floor('h')
    
    agg_funcs = {
        'sum': 'sum',
        'mean': 'mean',
        'max': 'max',
        'min': 'min',
        'p80': lambda x: np.percentile(x, 80)
    }
    
    agg_func = agg_funcs.get(method, 'sum')
    
    # Definir colunas de agrupamento
    group_by = ['Data_Hora']
    if group_cols:
        group_by.extend(group_cols)
    
    df_agg = df.groupby(group_by).agg({value_col: agg_func}).reset_index()
    
    # Renomear coluna de data
    df_agg = df_agg.rename(columns={'Data_Hora': 'Data'})
    
    return df_agg


# ====== INTERFACE ======

st.title("📁 Upload de Dados")

st.markdown("""
Faça upload do seu arquivo CSV ou Excel contendo dados de série temporal.
O arquivo deve conter pelo menos:
- Uma coluna de **data/hora**
- Uma coluna com os **valores** a serem previstos
""")

# Upload
uploaded_file = st.file_uploader(
    "Escolha um arquivo",
    type=['csv', 'xlsx', 'xls'],
    help="Formatos suportados: CSV, Excel (.xlsx, .xls)"
)

if uploaded_file is not None:
    # Detectar tipo e carregar
    try:
        if uploaded_file.name.endswith('.csv'):
            # Tentar diferentes separadores
            try:
                df_raw = pd.read_csv(uploaded_file, sep=';', encoding='utf-8')
                if len(df_raw.columns) == 1:
                    uploaded_file.seek(0)
                    df_raw = pd.read_csv(uploaded_file, sep=',', encoding='utf-8')
            except:
                uploaded_file.seek(0)
                df_raw = pd.read_csv(uploaded_file, encoding='latin-1')
        else:
            df_raw = pd.read_excel(uploaded_file)
        
        st.success(f"✅ Arquivo carregado: {len(df_raw)} linhas, {len(df_raw.columns)} colunas")
        
        # Preview
        with st.expander("📋 Preview dos dados brutos", expanded=True):
            st.dataframe(df_raw.head(10), width='stretch')
        
        st.markdown("---")
        st.markdown("### ⚙️ Configuração das Colunas")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Detectar colunas de data
            date_cols = [col for col in df_raw.columns 
                        if any(x in col.lower() for x in ['data', 'date', 'time', 'hora', 'dt'])]
            if not date_cols:
                date_cols = df_raw.columns.tolist()
            
            date_col = st.selectbox(
                "Coluna de Data/Hora",
                options=df_raw.columns.tolist(),
                index=df_raw.columns.tolist().index(date_cols[0]) if date_cols else 0
            )
        
        with col2:
            # Detectar colunas numéricas
            numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
            value_cols = [col for col in numeric_cols 
                         if any(x in col.lower() for x in ['demand', 'valor', 'value', 'qty', 'quant'])]
            if not value_cols:
                value_cols = numeric_cols
            
            value_col = st.selectbox(
                "Coluna de Valores (Target)",
                options=numeric_cols if numeric_cols else df_raw.columns.tolist(),
                index=numeric_cols.index(value_cols[0]) if value_cols and numeric_cols else 0
            )
        
        # Colunas opcionais: Local e Sublocal
        st.markdown("### 📍 Colunas de Agrupamento (Opcionais)")
        
        all_cols = ['(Nenhum)'] + df_raw.columns.tolist()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Detectar coluna Local
            local_cols = [col for col in df_raw.columns 
                         if any(x in col.lower() for x in ['local', 'loja', 'filial', 'unidade', 'site', 'location'])]
            default_local = local_cols[0] if local_cols else '(Nenhum)'
            
            local_col = st.selectbox(
                "Coluna de Local",
                options=all_cols,
                index=all_cols.index(default_local) if default_local in all_cols else 0,
                help="Coluna que identifica diferentes locais/lojas/filiais"
            )
        
        with col2:
            # Detectar coluna Sublocal
            sublocal_cols = [col for col in df_raw.columns 
                           if any(x in col.lower() for x in ['sublocal', 'setor', 'departamento', 'area', 'subloc'])]
            default_sublocal = sublocal_cols[0] if sublocal_cols else '(Nenhum)'
            
            sublocal_col = st.selectbox(
                "Coluna de Sublocal",
                options=all_cols,
                index=all_cols.index(default_sublocal) if default_sublocal in all_cols else 0,
                help="Coluna que identifica subníveis (setor, departamento, etc.)"
            )
        
        # Converter coluna de data
        df = df_raw.copy()
        try:
            df[date_col] = pd.to_datetime(df[date_col], dayfirst=True)
        except:
            try:
                df[date_col] = pd.to_datetime(df[date_col])
            except Exception as e:
                st.error(f"Erro ao converter data: {e}")
                st.stop()
        
        # Detectar se tem hora
        has_time = detect_datetime_has_time(df, date_col)
        
        st.markdown("---")
        st.markdown("### 📊 Granularidade dos Dados")
        
        if has_time:
            st.info("🕐 Dados com componente de hora detectados")
            
            granularidade = st.radio(
                "Selecione a granularidade desejada:",
                options=['Horária (manter original)', 'Diária (agregar)', 'Por Turnos'],
                horizontal=True
            )
            
            if granularidade != 'Horária (manter original)':
                metodo_agg = st.selectbox(
                    "Método de agregação:",
                    options=['Soma', 'Média', 'Máximo', 'Mínimo', 'Percentil 80%'],
                    index=0
                )
                
                metodo_map = {
                    'Soma': 'sum',
                    'Média': 'mean',
                    'Máximo': 'max',
                    'Mínimo': 'min',
                    'Percentil 80%': 'p80'
                }
                metodo = metodo_map[metodo_agg]
            
            # Configuração de turnos
            if granularidade == 'Por Turnos':
                st.markdown("#### ⏰ Configuração dos Turnos")
                
                # Escolher quantidade de turnos
                num_turnos = st.selectbox(
                    "Quantidade de turnos:",
                    options=[2, 3, 4, 5],
                    index=1,  # Default: 3 turnos
                    help="Selecione quantos turnos deseja configurar"
                )
                
                # Nomes padrão para os turnos
                nomes_padrao = {
                    2: ['Diurno', 'Noturno'],
                    3: ['Manhã', 'Tarde', 'Noite'],
                    4: ['Madrugada', 'Manhã', 'Tarde', 'Noite'],
                    5: ['Madrugada', 'Manhã', 'Almoço', 'Tarde', 'Noite']
                }
                
                # Horários padrão para cada configuração
                horas_padrao = {
                    2: [(6, 18), (18, 6)],
                    3: [(6, 14), (14, 22), (22, 6)],
                    4: [(0, 6), (6, 12), (12, 18), (18, 24)],
                    5: [(0, 6), (6, 11), (11, 14), (14, 19), (19, 24)]
                }
                
                turnos = {}
                
                # Criar colunas dinamicamente
                cols = st.columns(num_turnos)
                
                for i in range(num_turnos):
                    with cols[i]:
                        nome_default = nomes_padrao[num_turnos][i]
                        hora_ini_default, hora_fim_default = horas_padrao[num_turnos][i]
                        
                        # Nome do turno editável
                        nome_turno = st.text_input(
                            f"Nome do Turno {i+1}",
                            value=nome_default,
                            key=f'turno_nome_{i}'
                        )
                        
                        # Horário de início
                        t_inicio = st.number_input(
                            "Início (hora)", 
                            min_value=0, 
                            max_value=23, 
                            value=hora_ini_default if hora_ini_default < 24 else 0, 
                            key=f't{i}_ini'
                        )
                        
                        # Horário de fim
                        t_fim = st.number_input(
                            "Fim (hora)", 
                            min_value=0, 
                            max_value=23, 
                            value=hora_fim_default if hora_fim_default < 24 else 0, 
                            key=f't{i}_fim'
                        )
                        
                        turnos[nome_turno] = {'inicio': t_inicio, 'fim': t_fim}
                
                # Mostrar resumo dos turnos
                st.markdown("##### 📋 Resumo dos Turnos Configurados:")
                resumo_turnos = []
                for nome, config in turnos.items():
                    ini = config['inicio']
                    fim = config['fim']
                    if ini < fim:
                        periodo = f"{ini:02d}:00 - {fim:02d}:00"
                    else:
                        periodo = f"{ini:02d}:00 - {fim:02d}:00 (atravessa meia-noite)"
                    resumo_turnos.append(f"**{nome}**: {periodo}")
                
                st.info(" | ".join(resumo_turnos))
        else:
            st.info("📅 Dados com granularidade diária detectados")
            granularidade = 'Diária'
            
            # Verificar se há múltiplos registros por dia
            df_temp = df.copy()
            df_temp['_data_only'] = pd.to_datetime(df_temp[date_col]).dt.date
            registros_por_dia = df_temp.groupby('_data_only').size()
            tem_multiplos = registros_por_dia.max() > 1
            
            if tem_multiplos:
                st.warning(f"⚠️ Detectados múltiplos registros por dia (máx: {registros_por_dia.max()})")
                
                metodo_agg_diario = st.selectbox(
                    "Método de agregação diária:",
                    options=['Não agregar (manter original)', 'Soma', 'Média', 'Máximo', 'Mínimo', 'Percentil 80%'],
                    index=1,
                    help="Como agregar múltiplos registros do mesmo dia"
                )
                
                metodo_map_diario = {
                    'Não agregar (manter original)': None,
                    'Soma': 'sum',
                    'Média': 'mean',
                    'Máximo': 'max',
                    'Mínimo': 'min',
                    'Percentil 80%': 'p80'
                }
                metodo_diario = metodo_map_diario[metodo_agg_diario]
            else:
                metodo_diario = None
        
        # Processar dados
        st.markdown("---")
        
        if st.button("✅ Processar e Salvar Dados", type="primary"):
            with st.spinner("Processando dados..."):
                # Colunas a manter
                cols_to_keep = [date_col, value_col]
                if local_col != '(Nenhum)':
                    cols_to_keep.append(local_col)
                if sublocal_col != '(Nenhum)':
                    cols_to_keep.append(sublocal_col)
                
                # Montar group_cols para agregação mantendo Local/Sublocal
                group_cols = []
                if local_col != '(Nenhum)':
                    group_cols.append(local_col)
                if sublocal_col != '(Nenhum)':
                    group_cols.append(sublocal_col)
                
                # Aplicar agregação se necessário
                if has_time and granularidade == 'Diária (agregar)':
                    df_final = aggregate_to_daily(df, date_col, value_col, metodo, group_cols if group_cols else None)
                    # Renomear colunas de Local/Sublocal para nomes padrão
                    if local_col != '(Nenhum)' and local_col in df_final.columns:
                        df_final = df_final.rename(columns={local_col: 'Local'})
                    if sublocal_col != '(Nenhum)' and sublocal_col in df_final.columns:
                        df_final = df_final.rename(columns={sublocal_col: 'Sublocal'})
                    st.session_state.granularidade_tipo = 'diaria'
                    
                elif has_time and granularidade == 'Por Turnos':
                    df_final = aggregate_to_shifts(df, date_col, value_col, turnos, metodo, group_cols if group_cols else None)
                    # Renomear colunas de Local/Sublocal para nomes padrão
                    if local_col != '(Nenhum)' and local_col in df_final.columns:
                        df_final = df_final.rename(columns={local_col: 'Local'})
                    if sublocal_col != '(Nenhum)' and sublocal_col in df_final.columns:
                        df_final = df_final.rename(columns={sublocal_col: 'Sublocal'})
                    st.session_state.granularidade_tipo = 'turno'
                    
                elif has_time and granularidade == 'Horária (manter original)':
                    # Manter granularidade horária com Local/Sublocal
                    df_final = df[cols_to_keep].copy()
                    df_final[date_col] = pd.to_datetime(df_final[date_col])
                    
                    # Extrair hora como coluna separada para facilitar análise
                    df_final['Hora'] = df_final[date_col].dt.hour
                    
                    # Renomear colunas
                    rename_dict = {date_col: 'Data', value_col: 'Demanda'}
                    if local_col != '(Nenhum)':
                        rename_dict[local_col] = 'Local'
                    if sublocal_col != '(Nenhum)':
                        rename_dict[sublocal_col] = 'Sublocal'
                    df_final = df_final.rename(columns=rename_dict)
                    st.session_state.granularidade_tipo = 'horaria'
                    
                else:
                    # Dados diários (sem hora)
                    df_final = df[cols_to_keep].copy()
                    df_final[date_col] = pd.to_datetime(df_final[date_col])
                    
                    # Renomear colunas
                    rename_dict = {date_col: 'Data', value_col: 'Demanda'}
                    if local_col != '(Nenhum)':
                        rename_dict[local_col] = 'Local'
                    if sublocal_col != '(Nenhum)':
                        rename_dict[sublocal_col] = 'Sublocal'
                    df_final = df_final.rename(columns=rename_dict)
                    
                    # Aplicar agregação diária se selecionada
                    if 'metodo_diario' in dir() and metodo_diario is not None:
                        # Agregar por dia (e Local/Sublocal se existirem)
                        agg_cols = ['Data']
                        if 'Local' in df_final.columns:
                            agg_cols.append('Local')
                        if 'Sublocal' in df_final.columns:
                            agg_cols.append('Sublocal')
                        
                        if metodo_diario == 'p80':
                            df_final = df_final.groupby(agg_cols).agg({
                                'Demanda': lambda x: x.quantile(0.8)
                            }).reset_index()
                        else:
                            df_final = df_final.groupby(agg_cols).agg({
                                'Demanda': metodo_diario
                            }).reset_index()
                        
                        st.session_state.metodo_agregacao = metodo_diario
                    else:
                        st.session_state.metodo_agregacao = None
                    
                    st.session_state.granularidade_tipo = 'diaria'
                
                # Renomear coluna de valor para Demanda se ainda não foi
                if 'Demanda' not in df_final.columns and value_col in df_final.columns:
                    df_final = df_final.rename(columns={value_col: 'Demanda'})
                
                # Ordenar por data
                df_final = df_final.sort_values('Data').reset_index(drop=True)
                
                # Remover NaN na demanda
                df_final = df_final.dropna(subset=['Demanda'])
                
                # Salvar no session_state
                st.session_state.df = df_final
                st.session_state.dados_carregados = True
                st.session_state.target_col = 'Demanda'
                st.session_state.granularidade_salva = granularidade if has_time else 'Diária'
                
                st.success(f"✅ Dados processados: {len(df_final)} registros")
                
                # Mostrar preview
                st.markdown("### 📋 Dados Processados")
                st.dataframe(df_final.head(20), width='stretch')
                
                # Estatísticas
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Registros", f"{len(df_final):,}")
                with col2:
                    st.metric("Data Início", df_final['Data'].min().strftime('%d/%m/%Y'))
                with col3:
                    st.metric("Data Fim", df_final['Data'].max().strftime('%d/%m/%Y'))
                with col4:
                    st.metric("Média", f"{df_final['Demanda'].mean():,.2f}")
                
                # Gráfico básico (filtros disponíveis após recarregar a página)
                import plotly.express as px
                
                st.markdown("### 📈 Série Temporal")
                fig = px.line(df_final, x='Data', y='Demanda', title='Série Temporal Processada')
                st.plotly_chart(fig, width='stretch')
                
                st.info("👉 Vá para **Treinamento** para treinar os modelos, ou recarregue a página para usar os filtros do gráfico.")
    
    except Exception as e:
        st.error(f"❌ Erro ao carregar arquivo: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

else:
    # Mostrar dados carregados anteriormente
    if 'df' in st.session_state and st.session_state.df is not None:
        st.success("✅ Dados já carregados na sessão")
        
        df = st.session_state.df
        
        st.markdown("### 📋 Dados Atuais")
        st.dataframe(df.head(20), width='stretch')
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Registros", f"{len(df):,}")
        with col2:
            st.metric("Data Início", df['Data'].min().strftime('%d/%m/%Y'))
        with col3:
            st.metric("Data Fim", df['Data'].max().strftime('%d/%m/%Y'))
        with col4:
            st.metric("Média", f"{df['Demanda'].mean():,.2f}")
        
        # ========== GRÁFICO COM FILTROS INTERATIVOS ==========
        import plotly.express as px
        import plotly.graph_objects as go
        
        st.markdown("### 📈 Série Temporal")
        st.markdown("#### 🔍 Filtros do Gráfico")
        
        # Preparar opções de filtro
        locais_disponiveis = ['Todos']
        sublocais_disponiveis = ['Todos']
        turnos_disponiveis = ['Todos']
        
        # Extrair locais únicos
        if 'Local' in df.columns:
            locais_unicos = df['Local'].dropna().unique().tolist()
            locais_disponiveis.extend(sorted([str(l) for l in locais_unicos if str(l) != 'Todos']))
        
        # Extrair sublocais únicos
        if 'Sublocal' in df.columns:
            sublocais_unicos = df['Sublocal'].dropna().unique().tolist()
            sublocais_disponiveis.extend(sorted([str(s) for s in sublocais_unicos if str(s) != 'Todos']))
        
        # Verificar se há turnos
        tem_turnos = 'Turno' in df.columns
        if tem_turnos:
            turnos_unicos = df['Turno'].dropna().unique().tolist()
            turnos_disponiveis.extend(sorted([str(t) for t in turnos_unicos]))
        
        # Layout dos filtros - Primeira linha
        col_f1, col_f2, col_f3 = st.columns(3)
        
        with col_f1:
            filtro_local = st.selectbox(
                "📍 Local",
                locais_disponiveis,
                key="upload_filtro_local"
            )
        
        with col_f2:
            # Filtrar sublocais com base no local selecionado
            if filtro_local != 'Todos' and 'Local' in df.columns and 'Sublocal' in df.columns:
                sublocais_do_local = df[df['Local'] == filtro_local]['Sublocal'].dropna().unique().tolist()
                sublocais_filtrados = ['Todos'] + sorted([str(s) for s in sublocais_do_local if str(s) != 'Todos'])
            else:
                sublocais_filtrados = sublocais_disponiveis
            
            filtro_sublocal = st.selectbox(
                "📍 Sublocal",
                sublocais_filtrados,
                key="upload_filtro_sublocal"
            )
        
        with col_f3:
            if tem_turnos:
                filtro_turno = st.selectbox(
                    "🕐 Turno",
                    turnos_disponiveis,
                    key="upload_filtro_turno"
                )
            else:
                filtro_turno = 'Todos'
                st.info("Sem turnos nos dados")
        
        # Layout dos filtros - Segunda linha
        col_f4, col_f5 = st.columns(2)
        
        with col_f4:
            eixo_x_opcoes = ['Por Dia']
            if 'Hora' in df.columns:
                eixo_x_opcoes.append('Por Hora')
            
            eixo_x_grafico = st.selectbox(
                "📅 Eixo X",
                eixo_x_opcoes,
                key="upload_filtro_eixo_x"
            )
        
        with col_f5:
            agregador_grafico = st.selectbox(
                "📊 Agregador",
                ["Soma", "Média", "Máximo", "Mínimo", "Percentil 80%"],
                key="upload_filtro_agregador"
            )
        
        # Aplicar filtros
        df_filtrado = df.copy()
        
        if filtro_local != 'Todos' and 'Local' in df_filtrado.columns:
            # Comparar como string para evitar problemas de tipo
            df_filtrado = df_filtrado[df_filtrado['Local'].astype(str) == str(filtro_local)]
        
        if filtro_sublocal != 'Todos' and 'Sublocal' in df_filtrado.columns:
            # Comparar como string para evitar problemas de tipo
            df_filtrado = df_filtrado[df_filtrado['Sublocal'].astype(str) == str(filtro_sublocal)]
        
        if filtro_turno != 'Todos' and 'Turno' in df_filtrado.columns:
            # Comparar como string para evitar problemas de tipo
            df_filtrado = df_filtrado[df_filtrado['Turno'].astype(str) == str(filtro_turno)]
        
        # Função de agregação
        def aplicar_agregacao_upload(series, tipo):
            if tipo == "Soma":
                return series.sum()
            elif tipo == "Média":
                return series.mean()
            elif tipo == "Máximo":
                return series.max()
            elif tipo == "Mínimo":
                return series.min()
            elif tipo == "Percentil 80%":
                return series.quantile(0.8) if len(series) > 0 else 0
            return series.sum()
        
        # Preparar dados para o gráfico
        if eixo_x_grafico == 'Por Hora' and 'Hora' in df_filtrado.columns:
            # Agregação por Data + Hora
            df_agg = df_filtrado.groupby(['Data', 'Hora']).agg({
                'Demanda': lambda x: aplicar_agregacao_upload(x, agregador_grafico)
            }).reset_index()
            df_agg['DataHora'] = df_agg['Data'] + pd.to_timedelta(df_agg['Hora'], unit='h')
            df_agg = df_agg.sort_values('DataHora')
            x_col = 'DataHora'
            titulo_eixo_x = 'Data/Hora'
        else:
            # Agregação por Dia
            df_agg = df_filtrado.copy()
            df_agg['DataDia'] = df_agg['Data'].dt.date
            df_agg = df_agg.groupby('DataDia').agg({
                'Demanda': lambda x: aplicar_agregacao_upload(x, agregador_grafico)
            }).reset_index()
            df_agg['DataDia'] = pd.to_datetime(df_agg['DataDia'])
            df_agg = df_agg.sort_values('DataDia')
            x_col = 'DataDia'
            titulo_eixo_x = 'Data'
        
        # Título dinâmico
        titulo_filtros = []
        if filtro_local != 'Todos':
            titulo_filtros.append(f"Local: {filtro_local}")
        if filtro_sublocal != 'Todos':
            titulo_filtros.append(f"Sublocal: {filtro_sublocal}")
        if filtro_turno != 'Todos':
            titulo_filtros.append(f"Turno: {filtro_turno}")
        
        titulo_base = f'Série Temporal ({agregador_grafico})'
        if titulo_filtros:
            titulo_base += f" | {' | '.join(titulo_filtros)}"
        
        # Criar gráfico
        if len(df_agg) > 0:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_agg[x_col],
                y=df_agg['Demanda'],
                mode='lines',
                name='Demanda',
                line=dict(color='#1f77b4', width=2)
            ))
            
            fig.update_layout(
                title=titulo_base,
                xaxis_title=titulo_eixo_x,
                yaxis_title=f'Demanda ({agregador_grafico})',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, width='stretch')
        else:
            st.warning("⚠️ Nenhum dado encontrado com os filtros selecionados.")
        
        # ========== FIM DO GRÁFICO ==========
        
        if st.button("🗑️ Limpar dados e carregar novo arquivo"):
            st.session_state.df = None
            st.session_state.dados_carregados = False
            st.rerun()
    else:
        st.info("👆 Faça upload de um arquivo para começar")
