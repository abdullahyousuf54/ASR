from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asr_pipeline.config import ensure_project_dirs, load_config
from asr_pipeline.data import prepare_datasets, save_prepared_dataset
from asr_pipeline.utils import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Hindi/Hinglish ASR datasets.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_project_dirs(cfg)
    set_seed(int(cfg["project"].get("seed", 1337)))
    dataset_dict = prepare_datasets(cfg)
    save_prepared_dataset(cfg, dataset_dict)
    print(f"Prepared splits: {', '.join(f'{k}={len(v)}' for k, v in dataset_dict.items())}")


if __name__ == "__main__":
    main()

