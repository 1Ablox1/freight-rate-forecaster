"""Train a RandomForest regressor on synthetic freight rates."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "sample_rates.csv"
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "rate_model.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
MODEL_VERSION = "rf-v1"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df["dayofweek"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["days_from_start"] = (df["date"] - df["date"].min()).dt.days
    return df


def build_pipeline() -> Pipeline:
    categorical = ["origin", "dest", "mode"]
    numeric = ["dayofweek", "month", "days_from_start"]
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", "passthrough", numeric),
        ]
    )
    model = RandomForestRegressor(
        n_estimators=120,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("pre", pre), ("model", model)])


def train() -> dict:
    df = load_data()
    features = ["origin", "dest", "mode", "dayofweek", "month", "days_from_start"]
    x = df[features]
    y = df["rate"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    pipe = build_pipeline()
    pipe.fit(x_train, y_train)
    preds = pipe.predict(x_test)
    mae = float(mean_absolute_error(y_test, preds))
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    metrics = {
        "model_version": MODEL_VERSION,
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe, "model_version": MODEL_VERSION}, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    train()
