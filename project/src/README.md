# Исходный код ChurnGuard

- `config.py` - загрузка YAML-конфига и разрешение путей относительно папки проекта.
- `data.py` - генерация синтетического датасета и загрузка CSV.
- `model.py` - preprocessing pipeline, обучение, оценка, сохранение и загрузка модели.
- `train.py` - CLI-точка входа для обучения.
- `service.py` - FastAPI-сервис с `/health` и `/predict`.

Основные команды:

```bash
python -m src.train
python -m src.service
```
