import pandas as pd
from typing import Tuple

def load_price_data(
        csv_path:str,
        target_col:str="Close"
) -> pd.Series:
    df=pd.read_csv(csv_path,
                   parse_dates=["Date"],
                   index_col="Date").sort_index()
    series=df[target_col].astype(float)
    series.name = target_col
    return series

def train_test_split(
        series:pd.Series,
        split_date:str
)->Tuple[pd.Series,pd.Series]:
    train=series.loc[series.index<split_date]
    test=series.loc[series.index>=split_date]

    if len(test) == 0:
        raise ValueError("test set is empty.check the date")
    if len(train) == 0:
        raise ValueError("train set is empty,enter a valid split date")
    return train , test

