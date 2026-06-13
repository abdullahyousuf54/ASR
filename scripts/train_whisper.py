from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asr_pipeline.config import ensure_project_dirs, load_config
from asr_pipeline.training import train_whisper


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Whisper for Hindi/Hinglish ASR.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_project_dirs(cfg)
    result = train_whisper(cfg)
    print(result)


if __name__ == "__main__":
    main()

