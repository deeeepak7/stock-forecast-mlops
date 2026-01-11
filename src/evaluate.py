from sklearn.metrics import mean_absolute_error,mean_squared_error
import numpy as np

def metrics(y_true,y_pred)->dict:
    mae=mean_absolute_error(y_true,y_pred)
    mse=mean_squared_error(y_true,y_pred)
    rmse=np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return{
        "mae":mae,
        "rmse":rmse,
        "mape":mape
    }

