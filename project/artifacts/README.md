# Артефакты

Файлы создаются командой:

```bash
cd project
python -m src.train
```

- `churn_model.joblib` - финальная модель `RandomForestClassifier` вместе с preprocessing pipeline.
- `metrics.json` - метрики baseline и финальной модели на тестовой выборке.

Эти артефакты нужны для демонстрации: сервис `src.service` загружает `churn_model.joblib`, а отчет ссылается на `metrics.json`.
