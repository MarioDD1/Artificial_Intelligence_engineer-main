from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import resolve_project_path
from .data import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, NUMERIC_COLUMNS


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLUMNS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
        ]
    )


def build_pipeline(model_name: str, model_config: dict[str, Any], random_state: int) -> Pipeline:
    if model_name == "logistic_regression":
        estimator = LogisticRegression(max_iter=int(model_config.get("max_iter", 1000)), random_state=random_state)
    elif model_name == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=int(model_config.get("n_estimators", 160)),
            max_depth=int(model_config.get("max_depth", 8)),
            min_samples_leaf=int(model_config.get("min_samples_leaf", 4)),
            random_state=random_state,
            class_weight="balanced",
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])


def evaluate_model(model: Pipeline, features: pd.DataFrame, target: pd.Series) -> dict[str, float]:
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]
    return {
        "accuracy": round(float(accuracy_score(target, predictions)), 4),
        "precision": round(float(precision_score(target, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(target, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(target, predictions, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(target, probabilities)), 4),
    }


def train_models(data: pd.DataFrame, config: dict[str, Any]) -> tuple[Pipeline, dict[str, Any]]:
    target_column = config["model"]["target"]
    random_state = int(config["project"]["random_state"])

    features = data[FEATURE_COLUMNS]
    target = data[target_column]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=float(config["data"]["test_size"]),
        random_state=random_state,
        stratify=target,
    )

    results: dict[str, Any] = {}
    trained_models: dict[str, Pipeline] = {}
    for model_name in ["logistic_regression", "random_forest"]:
        pipeline = build_pipeline(model_name, config["model"][model_name], random_state)
        pipeline.fit(x_train, y_train)
        trained_models[model_name] = pipeline
        results[model_name] = evaluate_model(pipeline, x_test, y_test)

    final_name = config["model"]["final_model"]
    results["final_model"] = final_name
    results["test_size"] = float(config["data"]["test_size"])
    results["n_train"] = int(len(x_train))
    results["n_test"] = int(len(x_test))
    return trained_models[final_name], results


def save_model(model: Pipeline, path: str | Path) -> None:
    model_path = resolve_project_path(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)


def load_model(path: str | Path) -> Pipeline:
    return joblib.load(resolve_project_path(path))
