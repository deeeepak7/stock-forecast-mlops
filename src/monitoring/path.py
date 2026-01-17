from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MONITORING_DIR = PROJECT_ROOT / "data" / "monitoring"
PREDICTIONS_FILE = MONITORING_DIR / "predictions.csv"
