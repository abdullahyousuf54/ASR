from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asr_pipeline.config import ensure_project_dirs, load_config
from asr_pipeline.reporting import build_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate ASR metrics and write the experiment report.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_project_dirs(cfg)
    result = build_report(cfg)
    print(f"Summary CSV: {result['summary_csv']}")
    print(f"Report: {result['report_path']}")
    if result["best"]:
        best = result["best"]
        print(
            "Best-performing configuration: "
            f"{best['run_name']} | mode={best['mode']} | WER={best['wer']:.4f} | CER={best['cer']:.4f}"
        )
    else:
        print("Best-performing configuration: not available yet")


if __name__ == "__main__":
    main()

