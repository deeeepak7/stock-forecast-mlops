import mlflow
from typing import Tuple

from src.data_loader import load_price_data,train_test_split
from src.arima import train_arima,walk_forward_Arima,arima_order


from src.retraining.config import PRODUCTION_RMSE,CSV_PATH,SPLIT_DATE,EXPERIMENT_NAME,ARTIFACT_DIR

def retrain_evaluate()-> Tuple[bool,float]:
    """
    retrain model and decide whether to promote or not
    
    :return: Description
    :rtype: Tuple[bool, float]
    """
    series=load_price_data(CSV_PATH)
    train,test=train_test_split(series,SPLIT_DATE)

    

    _,metric=walk_forward_Arima(train,test)

    candidate_rmse=metric["rmse"]
    
    return candidate_rmse<PRODUCTION_RMSE,candidate_rmse

def main():
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="ARIMA_RETRAIN_CANDIDATE"):
        promote,candidate_rmse = retrain_evaluate()

        mlflow.log_param("model_type","ARIMA")
        mlflow.log_param("p",arima_order[0])
        mlflow.log_param("d",arima_order[1])
        mlflow.log_param("q",arima_order[2])
        mlflow.log_metric("candidate_rmse",candidate_rmse)
        mlflow.log_metric("production_rmse",PRODUCTION_RMSE)

        if promote:
            model_path = ARTIFACT_DIR/"arima_model.pkl"
            model_path.parent.mkdir(exist_ok=True)

            full_series=load_price_data(CSV_PATH)
            final_model=train_arima(full_series)

            import pickle
            with open(model_path,"wb") as f:
                pickle.dump(final_model,f)

            mlflow.log_artifact(str(model_path))
            mlflow.set_tag("model_stage","promoted")

            print("new model promoted")

        else:
            mlflow.set_tag("model_stage","rejected")
            print("new model promoted") 

if __name__=="__main__":
    main()                   



