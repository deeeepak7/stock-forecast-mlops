import pandas as pd

from datetime import date
from src.monitoring.path import PREDICTIONS_FILE




#attach actual price to past prediction and compute abs_error
def update_actual(
        actual_price:float,
        actual_date:date,
)->None:
    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError("log not found")
    df=pd.read_csv(PREDICTIONS_FILE)
    if "actual" not in df.columns:
        df["actual"]=pd.NA
    if "abs_error" not in df.columns:
        df["abs_error"]=pd.NA    
    df["timestamp"]=pd.to_datetime(df["timestamp"])
    df["pred_date"]=df["timestamp"].dt.date
    mask=(df["pred_date"]==actual_date)&(df["actual"].isna()|(df["actual"]==""))
    if mask.sum() == 0:
        return
    df.loc[mask,"actual"]=actual_price
    df.loc[mask,"abs_error"]=(
        (df.loc[mask,"prediction"]-actual_price).abs()
    )
    df=df.drop(columns=["pred_date"])
    df.to_csv(PREDICTIONS_FILE,index=False)

