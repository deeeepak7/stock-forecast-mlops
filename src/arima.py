import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from typing import Tuple
from src.evaluate import metrics

arima_order=(0,1,0)

def walk_forward_Arima(
        train:pd.Series,
        test:pd.Series
)->Tuple[pd.Series,dict]:
    history=train.tolist()
    prediction=[]
    for t in range(len(test)):
        model = ARIMA(history,order=arima_order)
        model_fit=model.fit()
        y_hat=model_fit.forecast(steps=1)[0]
        prediction.append(y_hat)
        history.append(test.iloc[t])
    preds=pd.Series(prediction,index=test.index)
    metrix = metrics(test.values,preds.values)
    return preds,metrix

def train_arima(series:pd.Series):
    model= ARIMA(series,order=arima_order)
    model_fit = model.fit()
    return model_fit
