import pandas as pd
from typing import Tuple
from src.monitoring.path import PREDICTIONS_FILE


def detect_drift(
        *,
        training_rmse:float,
        window:int=14,
        threshold_multipier:float=1.5,
        consecutive_days:int=3,
)->Tuple[bool,float]:
    #return drift detection and latest rolling rmse
    if not PREDICTIONS_FILE.exists():
        return False,float("nan")
    df = pd.read_csv(PREDICTIONS_FILE)
    if "abs_error" not in df.columns:
        return False,float("nan")
    df=df.dropna(subset=["abs_error"])
    if len(df)<window:
        return False,float("nan")
    
    df["sq_error"]=df["abs_error"]**2
    df["rolling_rmse"]=(
        df["sq_error"].rolling(window=window).mean().pow(0.5)
    )
    threshold=training_rmse*threshold_multipier
    recent=df["rolling_rmse"].tail(consecutive_days)
    drift_detect=(recent>threshold).all()
    latest_rmse=recent.iloc[-1]

    return drift_detect,latest_rmse
