from src.data import FEATURE_COLUMNS, generate_churn_data


def test_generate_churn_data_has_expected_columns():
    data = generate_churn_data(n_samples=50, random_state=1)
    assert list(data.columns) == FEATURE_COLUMNS + ["churn"]
    assert len(data) == 50
    assert set(data["churn"].unique()).issubset({0, 1})


def test_contract_type_values_are_supported():
    data = generate_churn_data(n_samples=100, random_state=2)
    assert set(data["contract_type"].unique()).issubset({"month-to-month", "one-year", "two-year"})
