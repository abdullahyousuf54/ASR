from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .utils import read_json


def _collect_metric_rows(metrics_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(metrics_dir.glob("*.json")):
        data = read_json(path)
        if "modes" not in data:
            continue
        for mode, payload in data["modes"].items():
            if "after" in payload:
                before = payload["before"]
                after = payload["after"]
                rows.append(
                    {
                        "run_name": data.get("run_name", path.stem),
                        "split": data.get("split", ""),
                        "dataset": "overall",
                        "mode": mode,
                        "stage": "llm_before",
                        "wer": before.get("wer"),
                        "cer": before.get("cer"),
                        "exact_match": before.get("exact_match"),
                        "metrics_file": str(path),
                    }
                )
                rows.append(
                    {
                        "run_name": data.get("run_name", path.stem),
                        "split": data.get("split", ""),
                        "dataset": "overall",
                        "mode": mode,
                        "stage": "llm_after",
                        "wer": after.get("wer"),
                        "cer": after.get("cer"),
                        "exact_match": after.get("exact_match"),
                        "metrics_file": str(path),
                    }
                )
            else:
                rows.append(
                    {
                        "run_name": data.get("run_name", path.stem),
                        "split": data.get("split", ""),
                        "dataset": "overall",
                        "mode": mode,
                        "stage": "asr",
                        "wer": payload.get("wer"),
                        "cer": payload.get("cer"),
                        "exact_match": payload.get("exact_match"),
                        "rtf": data.get("rtf"),
                        "num_samples": data.get("num_samples"),
                        "metrics_file": str(path),
                    }
                )
                for dataset, ds_metrics in data.get("by_dataset", {}).get(mode, {}).items():
                    if dataset == "overall":
                        continue
                    rows.append(
                        {
                            "run_name": data.get("run_name", path.stem),
                            "split": data.get("split", ""),
                            "dataset": dataset,
                            "mode": mode,
                            "stage": "asr",
                            "wer": ds_metrics.get("wer"),
                            "cer": ds_metrics.get("cer"),
                            "exact_match": ds_metrics.get("exact_match"),
                            "rtf": data.get("rtf"),
                            "num_samples": data.get("num_samples"),
                            "metrics_file": str(path),
                        }
                    )
    return rows


def build_report(cfg: dict[str, Any]) -> dict[str, Any]:
    metrics_dir = Path(cfg["paths"]["metrics_dir"])
    summaries_dir = Path(cfg["paths"]["summaries_dir"])
    reports_dir = Path(cfg["paths"]["reports_dir"])
    summaries_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    rows = _collect_metric_rows(metrics_dir)
    df = pd.DataFrame(rows)
    summary_csv = summaries_dir / "summary.csv"
    df.to_csv(summary_csv, index=False)

    best = None
    if not df.empty:
        indic = df[(df["mode"] == "indic") & (df["dataset"] == "overall") & (df["wer"].notna())].copy()
        if not indic.empty:
            best = indic.sort_values("wer").iloc[0].to_dict()

    report_path = reports_dir / "experiment_report.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Hindi / Hinglish ASR Experiment Report\n\n")
        f.write("## Datasets Used\n\n")
        for spec in cfg.get("datasets", []):
            f.write(
                f"- `{spec.get('id')}`: kind={spec.get('kind')}, "
                f"train={spec.get('use_for_train')}, eval={spec.get('use_for_eval')}. "
                f"{spec.get('access_note', '')}\n"
            )
        f.write("\n## Preprocessing\n\n")
        f.write("- Audio is decoded as mono 16 kHz and filtered by configured duration limits.\n")
        f.write("- Raw transcript text is retained as `transcript_raw`.\n")
        f.write("- Training labels use `transcript_indic` after Hindi-preserving Indic normalization.\n")
        f.write("- Empty, corrupt, missing, and over-long samples are skipped.\n\n")
        f.write("## Normalization Strategy\n\n")
        f.write(
            "Both Whisper-style and Indic-normalized metrics are reported. Whisper-style normalization "
            "can make WER look lower, while Indic normalization is more faithful for Hindi script details.\n\n"
        )
        f.write("## Summary Metrics\n\n")
        if df.empty:
            f.write("No metric files were found yet. Run baseline, training evaluation, and optional LLM correction.\n")
        else:
            f.write(df.to_markdown(index=False))
            f.write("\n")
        f.write("\n## Best Configuration\n\n")
        if best:
            f.write(
                f"Best Indic WER currently comes from `{best['run_name']}` "
                f"with WER={best['wer']:.4f} and CER={best['cer']:.4f}.\n"
            )
        else:
            f.write("No completed run is available yet.\n")
        f.write("\n## Recommendations\n\n")
        f.write(
            "For production Hindi/Hinglish ASR, start with the fine-tuned Whisper checkpoint selected by "
            "lowest Indic WER, keep Whisper-style metrics only for comparability, and enable LLM correction "
            "only if it improves WER/CER on code-mixed and number-heavy validation subsets without raising "
            "semantic drift during manual review.\n"
        )

    return {"summary_csv": str(summary_csv), "report_path": str(report_path), "best": best, "rows": rows}
