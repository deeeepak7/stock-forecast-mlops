import pickle
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI,HTTPException
from statsmodels.tsa.arima.model import ARIMAResults
from src.arima import arima_order
from api.schema import PredictionOutcome,PredictionRequest


app=FastAPI(
    title="ARIMA FORECAST API",
    description="one step ahead stock price prediction forecasting",
    version="1.0.0"
)

MODEL_PATH=Path("artifacts/arima_model.pkl")

if not MODEL_PATH.exists():
    raise RuntimeError("Model artifact not found.train model first")

with open(MODEL_PATH,"rb") as f:
    arima_model:ARIMAResults=pickle.load(f)

@app.get("/health")
def check():
    return {"status":"ok"}

@app.post("/predict",response_model=PredictionOutcome)
def predict(request:PredictionRequest):
    try:
        history=request.recent_prices
        model=arima_model.model
        refit=model.clone(endog=history).fit()
        forecast=refit.forecast(steps=1)[0]

        return PredictionOutcome(
            prediction=float(forecast),
            model_type="arima",
            arima_order=arima_order,
            timestamp=datetime.utcnow()
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"prediction failed:{str(e)}"
        )
    