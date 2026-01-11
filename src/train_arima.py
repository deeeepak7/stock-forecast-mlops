import json
import pickle
import mlflow
from pathlib import Path
from src.data_loader import train_test_split,load_price_data
from src.arima import walk_forward_Arima,arima_order,train_arima

#config
CSV_PATH="data/raw/aapl_2020_2024.csv"
SPLIT_DATE="2024-01-01"
EXPERIMENT_NAME = "ARIMA_PRODUCTION"

ARTIFACT_DIR = "artifacts"
Path(ARTIFACT_DIR).mkdir(exist_ok=True)

#mlflow
mlflow.set_experiment(EXPERIMENT_NAME)
with mlflow.start_run(run_name="ARIMA_0_1_0_FINAL"):
    #load data
    series=load_price_data(CSV_PATH)
    train,test=train_test_split(series,SPLIT_DATE)

    #walk forward evaluation 
    prediction,metrics=walk_forward_Arima(train,test)

    #train arima
    final_model = train_arima(series) # final model retrained on full historical data for deployment

    #log
    mlflow.log_param("model_type","ARIMA")
    mlflow.log_param("p",arima_order[0])
    mlflow.log_param("d",arima_order[1])
    mlflow.log_param("q",arima_order[2])
    mlflow.log_param("split_date",SPLIT_DATE)
    mlflow.log_param("forecast_horizon","1-step_ahead")

    #log metrics
    for k,v in metrics.items():
        mlflow.log_metric(k,v)


    #save artificat
    metrics_path=f"{ARTIFACT_DIR}/metrics.json"
    pred_path = f"{ARTIFACT_DIR}/predictions.json"
    model_path=f"{ARTIFACT_DIR}/arima_model.pkl"

    with open(metrics_path,"w") as f:
        json.dump(metrics,f,indent=4) 
    prediction.to_csv(pred_path)

    with open(model_path,"wb") as f:
        pickle.dump(final_model,f)

    mlflow.log_artifact(metrics_path)
    mlflow.log_artifact(pred_path)     
    mlflow.log_artifact(model_path)  