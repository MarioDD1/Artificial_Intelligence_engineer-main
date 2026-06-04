from fastapi.testclient import TestClient

from src.service import app


def test_health_and_predict_endpoints_work():
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    response = client.post(
        "/predict",
        json={
            "age": 34,
            "tenure_months": 8,
            "monthly_spend": 92.5,
            "support_tickets": 3,
            "late_payments": 1,
            "contract_type": "month-to-month",
            "has_autopay": 0,
            "uses_mobile_app": 1,
            "satisfaction_score": 5.2,
        },
    )
    payload = response.json()
    assert response.status_code == 200
    assert 0 <= payload["churn_probability"] <= 1
    assert payload["risk_level"] in {"low", "medium", "high"}
