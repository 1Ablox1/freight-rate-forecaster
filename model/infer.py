"""Load trained artifact and predict a freight rate."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "rate_model.joblib"
DATA_PATH = ROOT / "data" / "sample_rates.csv"

_bundle = None
_ref_start: pd.Timestamp | None = None


def _load():
    global _bundle, _ref_start
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Missing model at {MODEL_PATH}. Run: python -m model.train"
            )
        _bundle = joblib.load(MODEL_PATH)
    if _ref_start is None:
        dates = pd.read_csv(DATA_PATH, usecols=["date"], parse_dates=["date"])
        _ref_start = dates["date"].min()
    return _bundle, _ref_start


def predict_rate(origin: str, dest: str, mode: str, on_date: str | date) -> dict:
    bundle, ref_start = _load()
    if isinstance(on_date, str):
        dt = datetime.strptime(on_date, "%Y-%m-%d").date()
    else:
        dt = on_date
    ts = pd.Timestamp(dt)
    row = pd.DataFrame(
        [
            {
                "origin": origin,
                "dest": dest,
                "mode": mode,
                "dayofweek": ts.dayofweek,
                "month": ts.month,
                "days_from_start": (ts - ref_start).days,
            }
        ]
    )
    pred = float(bundle["pipeline"].predict(row)[0])
    return {
        "predicted_rate": round(pred, 2),
        "model_version": bundle["model_version"],
        "currency": "USD",
        "unit": "per_FEU",
    }
