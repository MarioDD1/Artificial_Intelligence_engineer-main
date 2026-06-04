from __future__ import annotations

import logging
import os
from functools import lru_cache

import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

from .config import load_config
from .data import FEATURE_COLUMNS
from .model import load_model


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("churnguard")

config = load_config(os.getenv("CONFIG_PATH", "configs/config.yaml"))
app = FastAPI(title="ChurnGuard API", version="1.0.0")


class CustomerFeatures(BaseModel):
    age: int = Field(ge=18, le=100)
    tenure_months: int = Field(ge=0, le=120)
    monthly_spend: float = Field(ge=0)
    support_tickets: int = Field(ge=0)
    late_payments: int = Field(ge=0)
    contract_type: str = Field(pattern="^(month-to-month|one-year|two-year)$")
    has_autopay: int = Field(ge=0, le=1)
    uses_mobile_app: int = Field(ge=0, le=1)
    satisfaction_score: float = Field(ge=1, le=10)


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: int
    risk_level: str
    model: str


@lru_cache(maxsize=1)
def get_model():
    model_path = os.getenv("MODEL_PATH", config["model"]["artifact_path"])
    return load_model(model_path)


@app.get("/health")
def health() -> dict[str, str]:
    get_model()
    return {"status": "ok", "model": config["model"]["final_model"]}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures) -> PredictionResponse:
    frame = pd.DataFrame([features.model_dump()])[FEATURE_COLUMNS]
    probability = float(get_model().predict_proba(frame)[0, 1])
    threshold = float(config["service"]["decision_threshold"])
    prediction = int(probability >= threshold)
    if probability >= 0.7:
        risk_level = "high"
    elif probability >= 0.4:
        risk_level = "medium"
    else:
        risk_level = "low"

    logger.info("prediction probability=%.4f risk=%s", probability, risk_level)
    return PredictionResponse(
        churn_probability=round(probability, 4),
        churn_prediction=prediction,
        risk_level=risk_level,
        model=config["model"]["final_model"],
    )


def main() -> None:
    uvicorn.run(
        "src.service:app",
        host=config["service"]["host"],
        port=int(config["service"]["port"]),
        reload=False,
    )


if __name__ == "__main__":
    main()
