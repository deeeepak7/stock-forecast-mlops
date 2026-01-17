import csv
from datetime import datetime
from typing import Tuple
from src.monitoring.path import MONITORING_DIR,PREDICTIONS_FILE

def log_prediction(
        *,
        timestamp:datetime,
        prediction:float,
        history_len:int,
        model_type:str,
        arima_order:Tuple[int,int,int],
)->None:
    try:
        MONITORING_DIR.mkdir(exist_ok=True)
        file_exists=PREDICTIONS_FILE.exists()
        with open(PREDICTIONS_FILE,mode="a",newline="") as f:
            writer=csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "timestamp",
                    "prediction",
                    "actual",
                    "abs_error",
                    "history_len",
                    "model_type",
                    "arima_order"
                ])
            writer.writerow([
                timestamp.isoformat(),
                prediction,
                "",
                "",
                history_len,
                model_type,
                str(arima_order)
            ])    
    except Exception:
        pass
