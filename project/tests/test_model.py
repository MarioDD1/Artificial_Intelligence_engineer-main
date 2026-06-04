from src.config import load_config
from src.data import generate_churn_data
from src.model import train_models


def test_train_models_returns_final_model_and_metrics():
    config = load_config()
    data = generate_churn_data(n_samples=180, random_state=3)

    model, metrics = train_models(data, config)

    assert hasattr(model, "predict_proba")
    assert metrics["final_model"] == config["model"]["final_model"]
    assert "roc_auc" in metrics["random_forest"]
    assert 0 <= metrics["random_forest"]["roc_auc"] <= 1
