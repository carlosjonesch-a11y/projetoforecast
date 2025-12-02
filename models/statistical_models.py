"""
Modelos estatísticos.
"""

import numpy as np


class ARIMAModel:
    """Wrapper para ARIMA."""
    
    def __init__(self, order=(5, 1, 0)):
        self.order = order
        self.model = None
    
    def fit(self, y):
        from statsmodels.tsa.arima.model import ARIMA
        self.model = ARIMA(y, order=self.order).fit()
        return self
    
    def predict(self, steps):
        return self.model.forecast(steps=steps)


class ProphetModel:
    """Wrapper para Prophet."""
    
    def __init__(self, **kwargs):
        from prophet import Prophet
        self.model = Prophet(
            yearly_seasonality=kwargs.get('yearly_seasonality', True),
            weekly_seasonality=kwargs.get('weekly_seasonality', True),
            daily_seasonality=kwargs.get('daily_seasonality', False)
        )
    
    def fit(self, df):
        """df deve ter colunas 'ds' e 'y'."""
        self.model.fit(df)
        return self
    
    def predict(self, periods):
        future = self.model.make_future_dataframe(periods=periods)
        forecast = self.model.predict(future)
        return forecast['yhat'].tail(periods).values
