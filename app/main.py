"""FastAPI service for freight rate prediction."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from model.infer import predict_rate

app = FastAPI(
    title="Freight Rate Forecaster",
    description="Portfolio demo: predict freight rates from lane + date features",
    version="1.0.0",
)


class PredictRequest(BaseModel):
    origin: str = Field(..., examples=["CNSHA"])
    dest: str = Field(..., examples=["USLAX"])
    mode: str = Field(..., examples=["ocean"])
    date: str = Field(..., examples=["2025-06-15"], description="YYYY-MM-DD")


class PredictResponse(BaseModel):
    predicted_rate: float
    model_version: str
    currency: str
    unit: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest) -> PredictResponse:
    try:
        result = predict_rate(body.origin, body.dest, body.mode, body.date)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PredictResponse(**result)
