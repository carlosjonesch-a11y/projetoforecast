# 📈 Forecast Dashboard

Dashboard interativo para previsão de séries temporais com 12 modelos de Machine Learning e estatísticos.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🚀 Funcionalidades

- **📁 Upload de Dados**: Carregue arquivos CSV com validação automática
- **🤖 12 Modelos de Previsão**: XGBoost, LightGBM, CatBoost, Prophet, SARIMA, TBATS e mais
- **⚙️ Feature Engineering Automático**: 70+ features criadas automaticamente
- **🔍 GridSearchCV**: Otimização automática de hiperparâmetros
- **🎯 Ensemble Inteligente**: Combinação ponderada dos melhores modelos
- **📊 Comparativo de Métricas**: MAPE, RMSE, MAE, R² e mais
- **🔮 Previsões Futuras**: Com intervalos de confiança (80%, 95%)
- **📥 Exportação**: CSV e Excel para todos os resultados

## 📦 Instalação

### Requisitos

- Python 3.9+
- pip

### Instalação Local

```bash
# Clonar repositório
git clone https://github.com/seu-usuario/forecast-dashboard.git
cd forecast-dashboard

# Criar ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Instalar dependências
pip install -r requirements.txt

# Executar
streamlit run app.py
```

### Com Docker

```bash
docker build -t forecast-dashboard .
docker run -p 8501:8501 forecast-dashboard
```

## 🖥️ Uso

1. **Acesse o dashboard** em `http://localhost:8501`

2. **Upload de Dados**:
   - Vá para a página "📁 Upload Dados"
   - Carregue um arquivo CSV com colunas `Data` e `Demanda`
   - Formatos de data aceitos: `dd/mm/yyyy`, `yyyy-mm-dd`

3. **Treinamento**:
   - Acesse "🤖 Treinamento"
   - Selecione os modelos desejados
   - Configure parâmetros (GridSearchCV, threshold MAPE)
   - Clique em "Iniciar Treinamento"

4. **Visualização**:
   - "📊 Dashboard": KPIs e visão geral
   - "📈 Comparativo": Compare métricas entre modelos
   - "🔮 Previsões": Gere previsões futuras

5. **Exportação**:
   - Baixe previsões em CSV ou Excel
   - Exporte relatórios de métricas

## 🤖 Modelos Disponíveis

### Machine Learning

| Modelo | Descrição |
|--------|-----------|
| XGBoost | Gradient boosting otimizado |
| LightGBM | Gradient boosting rápido |
| CatBoost | Suporte a features categóricas |
| GradientBoosting | Sklearn baseline |
| RandomForest | Ensemble de árvores |
| MLPRegressor | Rede neural multicamadas |
| Ridge | Regressão linear regularizada |

### Estatísticos/Séries Temporais

| Modelo | Descrição |
|--------|-----------|
| ARIMA | Autoregressivo integrado |
| SARIMA | ARIMA com sazonalidade |
| Prophet | Modelo do Meta |
| TBATS | Múltiplas sazonalidades |
| Holt-Winters | Exponential smoothing |

## 📊 Métricas de Avaliação

- **MAPE**: Mean Absolute Percentage Error
- **SMAPE**: Symmetric MAPE
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **MASE**: Mean Absolute Scaled Error
- **R²**: Coeficiente de Determinação

## 📁 Estrutura do Projeto

```
forecast-dashboard/
├── .streamlit/
│   └── config.toml          # Configurações do Streamlit
├── app.py                    # Aplicação principal
├── pages/
│   ├── 1_📊_Dashboard.py     # KPIs e visão geral
│   ├── 2_📁_Upload_Dados.py  # Upload e validação
│   ├── 3_🤖_Treinamento.py   # Treinamento de modelos
│   ├── 4_🔮_Previsoes.py     # Previsões futuras
│   ├── 5_📈_Comparativo.py   # Métricas comparativas
│   └── 6_⚙️_Configuracoes.py # Configurações
├── models/
│   ├── __init__.py
│   ├── base_model.py         # Classe base abstrata
│   ├── ml_models.py          # Modelos ML
│   ├── statistical_models.py # Modelos estatísticos
│   └── ensemble.py           # Ensemble ponderado
├── utils/
│   ├── __init__.py
│   ├── data_loader.py        # Carregamento de dados
│   ├── preprocessing.py      # Feature engineering
│   ├── metrics.py            # Métricas de avaliação
│   └── helpers.py            # Funções auxiliares
├── visualization/
│   ├── __init__.py
│   ├── charts.py             # Gráficos Plotly
│   └── components.py         # Componentes UI
├── data/
│   └── sample_data.csv       # Dados de exemplo
├── requirements.txt          # Dependências Python
├── packages.txt              # Dependências sistema
└── README.md
```

## ☁️ Deploy no Streamlit Cloud

1. **Fork este repositório** para sua conta GitHub

2. **Acesse** [share.streamlit.io](https://share.streamlit.io)

3. **Configure o deploy**:
   - Repository: `seu-usuario/forecast-dashboard`
   - Branch: `main`
   - Main file path: `app.py`

4. **Aguarde o deploy** (pode levar alguns minutos na primeira vez)

### Configurações para Streamlit Cloud

O arquivo `packages.txt` já inclui dependências de sistema necessárias:
- `libgomp1` - OpenMP para paralelização
- `build-essential` - Ferramentas de compilação

## ⚙️ Configuração

### Parâmetros Principais

| Parâmetro | Descrição | Padrão |
|-----------|-----------|--------|
| `horizon` | Dias de previsão | 30 |
| `test_size` | Dias para teste | 30 |
| `mape_threshold` | Threshold para ensemble | 7% |
| `confidence_level` | Nível de confiança | 95% |
| `use_grid_search` | Otimização automática | True |

### Formato de Dados

O CSV deve conter:
- Coluna de **data**: `Data`, `Date`, `ds`, `Periodo`
- Coluna de **valor**: `Demanda`, `Volume`, `Vendas`, `y`

Exemplo:
```csv
Data,Demanda
01/01/2024,150
02/01/2024,175
03/01/2024,163
```

## 🔧 Desenvolvimento

### Executar Testes

```bash
pytest tests/ -v
```

### Linting

```bash
flake8 .
black .
```

## 📝 Changelog

### v1.0.0 (2024-11)
- Lançamento inicial
- 12 modelos de previsão
- Feature engineering automático
- Ensemble ponderado
- Deploy Streamlit Cloud

## 🤝 Contribuição

1. Fork o projeto
2. Crie uma branch (`git checkout -b feature/nova-funcionalidade`)
3. Commit suas mudanças (`git commit -m 'Add nova funcionalidade'`)
4. Push para a branch (`git push origin feature/nova-funcionalidade`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

## 🙏 Agradecimentos

- [Streamlit](https://streamlit.io/) - Framework web
- [Scikit-learn](https://scikit-learn.org/) - Machine Learning
- [Prophet](https://facebook.github.io/prophet/) - Modelo de séries temporais
- [Plotly](https://plotly.com/) - Visualizações interativas

---

Desenvolvido com ❤️ usando Streamlit
