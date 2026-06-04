from __future__ import annotations

import json

from .config import load_config, resolve_project_path
from .data import ensure_dataset
from .model import save_model, train_models


def main() -> None:
    config = load_config()
    data = ensure_dataset()
    model, metrics = train_models(data, config)

    save_model(model, config["model"]["artifact_path"])
    metrics_path = resolve_project_path(config["model"]["metrics_path"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Training complete")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
