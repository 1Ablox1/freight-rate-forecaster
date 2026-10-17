# Freight Rate Forecaster

Built in 2026 as a portfolio project. Demonstrates skills used in my logistics / automotive AI roles.

Theme: logistics pricing AI — predict lane rates from historical features.

> **Honest note:** `data/sample_rates.csv` is **synthetic** for demo purposes, not real carrier tariffs.

## Stack

- Python 3.11+, Pandas, scikit-learn (`RandomForestRegressor`)
- FastAPI `POST /predict`
- Artifacts saved under `artifacts/`

## Holdout metrics (after `python -m model.train`)

Metrics are written to `artifacts/metrics.json` (MAE / RMSE on an 80/20 split). Typical demo run on the bundled synthetic data:

| Metric | Value (synthetic holdout) |
|--------|---------------------------|
| MAE    | 37.379 USD                |
| RMSE   | 53.825 USD                |

Re-train locally and paste the printed JSON into PRs if you refresh the dataset.

## Quick start

```bash
cd 02-freight-rate-forecaster
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/generate_rates.py
python -m model.train
uvicorn app.main:app --reload --port 8002
```

Example:

```bash
curl -X POST http://127.0.0.1:8002/predict -H "Content-Type: application/json" -d "{\"origin\":\"CNSHA\",\"dest\":\"USLAX\",\"mode\":\"ocean\",\"date\":\"2025-06-15\"}"
```

## License

MIT — Abdalla Abdelsamie (2026).
