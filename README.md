# Stock Forecasting MLOps System

A production-grade, end-to-end MLOps system for stock price forecasting that encompasses the complete machine learning lifecycle—from model training and deployment to monitoring, drift detection, automated retraining, and CI/CD integration.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [ML Lifecycle](#ml-lifecycle)
- [Getting Started](#getting-started)
- [API Documentation](#api-documentation)
- [Monitoring & Operations](#monitoring--operations)
- [CI/CD Pipeline](#cicd-pipeline)
- [Docker Deployment](#docker-deployment)

## Overview

This system provides a robust solution for one-step-ahead stock closing price predictions using time-series forecasting models. The system exposes predictions through a REST API and implements comprehensive monitoring, automated drift detection, and safe model retraining workflows.

### Key Capabilities

- **Production-Ready ML Inference API**: FastAPI-based REST service for real-time predictions
- **Containerized Deployment**: Dockerized application ready for cloud deployment
- **Delayed-Label Monitoring**: Tracks predictions and updates with actual values via scheduled ETL jobs
- **Drift Detection**: Automated detection of model performance degradation using rolling RMSE metrics
- **Safe Model Retraining**: Automated retraining pipeline with validation and safe promotion mechanisms
- **CI/CD Automation**: GitHub Actions workflows for continuous integration and scheduled retraining

## Features

- ✅ Real-time stock price forecasting via REST API
- ✅ Model performance monitoring with delayed-label tracking
- ✅ Automated drift detection using rolling RMSE thresholds
- ✅ Safe model retraining and promotion workflows
- ✅ MLflow experiment tracking and model versioning
- ✅ Docker containerization for consistent deployments
- ✅ CI/CD pipelines for automated testing and retraining
- ✅ Production-grade error handling and logging

## Architecture

```
┌─────────┐
│ Client  │
└────┬────┘
     │ POST /predict
     ▼
┌─────────────────────────┐
│ FastAPI Inference       │
│ Service                 │
└────┬────────────────────┘
     │
     ├──► Prediction Logger
     │    └──► predictions.csv
     │
     │         (Daily ETL Job)
     │              │
     │              ▼
     │    ┌──────────────────┐
     │    │ Actuals Updater  │
     │    └────────┬─────────┘
     │             │
     │             ▼
     │    ┌──────────────────┐
     │    │ Error & Drift    │
     │    │ Detection        │
     │    └────────┬─────────┘
     │             │
     │    (if drift detected)
     │             │
     │             ▼
     │    ┌──────────────────┐
     │    │ Retraining       │
     │    │ Pipeline         │
     │    └────────┬─────────┘
     │             │
     │    (if performance improved)
     │             │
     │             ▼
     │    ┌──────────────────┐
     │    │ Model Promotion  │
     │    └──────────────────┘
```

## Technology Stack

| Category | Technology |
|----------|-----------|
| **Language** | Python 3.10+ |
| **ML Framework** | Statsmodels (ARIMA) |
| **API Framework** | FastAPI, Pydantic |
| **Experiment Tracking** | MLflow |
| **Monitoring** | Custom rolling RMSE, delayed-label tracking |
| **Data Source** | yfinance |
| **Containerization** | Docker |
| **CI/CD** | GitHub Actions |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |

## Project Structure

```
.
├── api/
│   ├── __init__.py
│   ├── main.py              # FastAPI inference service
│   └── schema.py            # Pydantic request/response models
│
├── src/
│   ├── arima.py             # ARIMA training & walk-forward evaluation
│   ├── data_loader.py       # Data loading utilities
│   ├── train_arima.py       # Model training script
│   ├── evaluate.py          # Model evaluation utilities
│   │
│   ├── monitoring/
│   │   ├── prediction_logger.py    # Log predictions to CSV
│   │   ├── actual_updater.py       # Update predictions with actuals
│   │   ├── drift.py                # Drift detection logic
│   │   └── fetch_actual.py         # Fetch actual stock prices
│   │
│   └── retraining/
│       ├── config.py        # Retraining configuration
│       └── retrain.py       # Retraining & safe promotion logic
│
├── data/
│   └── raw/
│       └── aapl_2020_2024.csv    # Historical stock data
│
├── artifacts/
│   ├── arima_model.pkl      # Production model artifact
│   ├── metrics.json         # Model performance metrics
│   └── predictions.json     # Prediction results
│
├── notebook/
│   ├── 1_eda.ipynb          # Exploratory data analysis
│   ├── 2_baseline.ipynb     # Baseline model experiments
│   ├── 3_arima.ipynb        # ARIMA model development
│   └── 4_lstm.ipynb         # LSTM model experiments
│
├── docker/
│   └── Dockerfile           # Docker configuration
│
├── .github/
│   └── workflows/
│       ├── ci.yml           # Continuous integration workflow
│       └── retrain.yml      # Scheduled retraining workflow
│
├── requirements.txt         # Python dependencies
└── README.md               # Project documentation
```

## ML Lifecycle

### 1. Model Training & Evaluation

- ARIMA model trained on historical stock price data
- Walk-forward validation for robust performance assessment
- Metrics tracked: MAE, RMSE, MAPE
- Experiments logged and versioned using MLflow

### 2. Inference API

- FastAPI service exposes `/predict` endpoint
- Accepts recent closing prices as input
- Returns one-step-ahead forecast with metadata
- Model loaded once at application startup for optimal performance

### 3. Deployment

- Application containerized using Docker
- Deployed to cloud platform (Render)
- Publicly accessible REST endpoint
- Health check endpoint available at `/health`

### 4. Monitoring (Delayed Labels)

- Predictions logged in real-time with timestamps
- Actual stock prices fetched via scheduled ETL job
- Prediction errors computed once actual values are available
- Performance metrics tracked over time

### 5. Drift Detection

- Rolling RMSE computed over recent predictions
- Drift detected when RMSE exceeds threshold for consecutive days
- Automated alerts trigger retraining pipeline

### 6. Retraining & Safe Promotion

- Model retrained offline on updated data
- New model evaluated using walk-forward validation
- Promotion occurs only if new model improves RMSE over production model
- Model versioning and promotion tracked via MLflow

### 7. CI/CD Automation

- **CI**: Docker build verification on every push
- **CD**: Scheduled retraining via GitHub Actions cron jobs
- Zero manual intervention required for model updates

## Getting Started

### Prerequisites

- Python 3.10 or higher
- pip package manager
- Docker (optional, for containerized deployment)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd stocky
   ```

2. **Create virtual environment**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the initial model** (if not already trained)
   ```bash
   python -m src.train_arima
   ```

5. **Run the API server**
   ```bash
   uvicorn api.main:app --reload
   ```

6. **Access the API documentation**
   - Interactive API docs: http://localhost:8000/docs
   - Alternative docs: http://localhost:8000/redoc

## API Documentation

### Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "ok"
}
```

### Prediction Endpoint

```http
POST /predict
```

**Request Body:**
```json
{
  "recent_prices": [
    170.2, 171.0, 169.8, 170.5, 171.3,
    172.1, 171.8, 172.6, 173.0, 172.4
  ]
}
```

**Response:**
```json
{
  "prediction": 173.45,
  "model_type": "arima",
  "arima_order": "(2, 1, 2)",
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

**Example using cURL:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "recent_prices": [170.2, 171.0, 169.8, 170.5, 171.3, 172.1, 171.8, 172.6, 173.0, 172.4]
  }'
```

## Monitoring & Operations

### Fetch Actual Prices

Update predictions with actual stock prices from the market:

```bash
python -m src.monitoring.fetch_actual
```

### Detect Drift

Check for model performance degradation:

```python
from src.monitoring.drift import detect_drift

# Detect drift using training RMSE as baseline
drift_detected = detect_drift(training_rmse=1.53)
```

### Retrain Model

Manually trigger model retraining:

```bash
python -m src.retraining.retrain
```

## CI/CD Pipeline

### Continuous Integration (`ci.yml`)

- Triggers on every push to the repository
- Validates Docker build
- Ensures code quality and deployment readiness

### Continuous Deployment (`retrain.yml`)

- Scheduled retraining via cron job
- Manual trigger support for on-demand retraining
- Automated model evaluation and promotion

## Docker Deployment

### Build Docker Image

```bash
docker build -t stock-forecast-api .
```

### Run Container

```bash
docker run -p 8000:8000 stock-forecast-api
```

### Docker Compose (Optional)

For more complex deployments, you can use Docker Compose to orchestrate multiple services.

## License

This project is provided as-is for educational and demonstration purposes.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please open an issue on the GitHub repository.

---

**Note**: This system is designed for educational and demonstration purposes. For production use, additional considerations such as authentication, rate limiting, and enhanced security measures should be implemented.
