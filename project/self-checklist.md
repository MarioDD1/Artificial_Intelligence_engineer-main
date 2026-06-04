# Самопроверка проекта

| # | Критерий | Да/Нет | Где смотреть / комментарий |
|---|---|---|---|
| 1 | Сервис запускается по инструкциям из `project/README.md` и работает | Да | `README.md`, раздел «Как запустить проект», `src/service.py` |
| 2 | Endpoint `/predict` использует реальную модель, а не заглушку | Да | `src/service.py`, `src/model.py`, `artifacts/churn_model.joblib` |
| 3 | Есть EDA и хотя бы один эксперимент с метриками | Да | `notebooks/01_eda_and_experiments.ipynb`, `artifacts/metrics.json`, `report.md` |
| 4 | Есть baseline и улучшенная модель, есть сравнение по метрикам | Да | `src/model.py`, `artifacts/metrics.json`, `report.md` |
| 5 | Код не свален в один ноутбук: есть структура в `src/` | Да | `src/data.py`, `src/model.py`, `src/train.py`, `src/service.py` |
| 6 | Есть Dockerfile или понятный сценарий развертывания без Docker | Да | `README.md`, команды `python -m src.train` и `python -m src.service` |
| 7 | Есть `.env.example` и нет реальных секретов/паролей | Да | `configs/.env.example`; проект не использует секреты |
| 8 | Реализованы логи/наблюдаемость: консольные логи и `/health` | Да | `src/service.py`, endpoint `/health` |
| 9 | В `report.md` обоснован выбор финальной модели | Да | `report.md`, разделы 5 и 8 |
| 10 | `README.md` и `report.md` позволяют понять сценарий демонстрации | Да | `README.md`, раздел 7; `report.md`, раздел 9 |

Итого: 10 из 10 пунктов закрыты на уровне учебного мини-проекта. Окончательная оценка зависит от проверки преподавателем и качества защиты.
