from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asr_pipeline.config import ensure_project_dirs, load_config
from asr_pipeline.evaluation import evaluate_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate pretrained Whisper baseline.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--run-name", default="whisper_baseline")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_project_dirs(cfg)
    result = evaluate_model(cfg, checkpoint=None, run_name=args.run_name, split=args.split)
    print(result["metrics"])


if __name__ == "__main__":
    main()

