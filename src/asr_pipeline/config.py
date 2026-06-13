from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(config_path)
    return cfg


def ensure_project_dirs(cfg: dict[str, Any]) -> None:
    for key in (
        "data_root",
        "processed_dir",
        "outputs_dir",
        "checkpoints_dir",
        "predictions_dir",
        "metrics_dir",
        "summaries_dir",
        "reports_dir",
    ):
        value = cfg.get("paths", {}).get(key)
        if value:
            Path(value).mkdir(parents=True, exist_ok=True)

