from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config, resolve_project_path


FEATURE_COLUMNS = [
    "age",
    "tenure_months",
    "monthly_spend",
    "support_tickets",
    "late_payments",
    "contract_type",
    "has_autopay",
    "uses_mobile_app",
    "satisfaction_score",
]

CATEGORICAL_COLUMNS = ["contract_type"]
NUMERIC_COLUMNS = [column for column in FEATURE_COLUMNS if column not in CATEGORICAL_COLUMNS]


def generate_churn_data(n_samples: int, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    age = rng.integers(18, 76, size=n_samples)
    tenure_months = rng.integers(1, 73, size=n_samples)
    monthly_spend = np.clip(rng.normal(65, 22, size=n_samples), 15, 160).round(2)
    support_tickets = rng.poisson(1.2, size=n_samples)
    late_payments = rng.poisson(0.45, size=n_samples)
    contract_type = rng.choice(["month-to-month", "one-year", "two-year"], size=n_samples, p=[0.55, 0.30, 0.15])
    has_autopay = rng.binomial(1, 0.58, size=n_samples)
    uses_mobile_app = rng.binomial(1, 0.66, size=n_samples)
    satisfaction_score = np.clip(rng.normal(7.1, 1.8, size=n_samples), 1, 10).round(1)

    contract_risk = np.select(
        [contract_type == "month-to-month", contract_type == "one-year", contract_type == "two-year"],
        [0.9, -0.25, -0.75],
    )
    linear_score = (
        -1.1
        + 0.42 * support_tickets
        + 0.58 * late_payments
        + 0.012 * (monthly_spend - 65)
        - 0.025 * tenure_months
        - 0.20 * has_autopay
        - 0.17 * uses_mobile_app
        - 0.38 * (satisfaction_score - 7)
        + contract_risk
        + rng.normal(0, 0.55, size=n_samples)
    )
    churn_probability = 1 / (1 + np.exp(-linear_score))
    churn = rng.binomial(1, churn_probability)

    return pd.DataFrame(
        {
            "age": age,
            "tenure_months": tenure_months,
            "monthly_spend": monthly_spend,
            "support_tickets": support_tickets,
            "late_payments": late_payments,
            "contract_type": contract_type,
            "has_autopay": has_autopay,
            "uses_mobile_app": uses_mobile_app,
            "satisfaction_score": satisfaction_score,
            "churn": churn,
        }
    )


def ensure_dataset(config_path: str | Path = "configs/config.yaml") -> pd.DataFrame:
    config = load_config(config_path)
    train_path = resolve_project_path(config["data"]["train_path"])
    sample_path = resolve_project_path(config["data"]["sample_path"])

    if train_path.exists():
        return pd.read_csv(train_path)

    train_path.parent.mkdir(parents=True, exist_ok=True)
    data = generate_churn_data(
        n_samples=int(config["data"]["n_samples"]),
        random_state=int(config["project"]["random_state"]),
    )
    data.to_csv(train_path, index=False)
    data.head(20).to_csv(sample_path, index=False)
    return data


def load_dataset(config_path: str | Path = "configs/config.yaml") -> pd.DataFrame:
    return ensure_dataset(config_path)


if __name__ == "__main__":
    dataset = ensure_dataset()
    print(f"Saved dataset with shape={dataset.shape}")
