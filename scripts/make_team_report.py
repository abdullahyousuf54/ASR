from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
METRICS_DIR = ROOT / "outputs" / "metrics"



def load_metrics(runs: list[dict]) -> list[dict]:
    rows = []
    for run in runs:
        if not run["path"].exists():
            raise FileNotFoundError(f"Missing metrics file: {run['path']}")
        data = json.loads(run["path"].read_text(encoding="utf-8"))
        for mode, metrics in data["modes"].items():
            rows.append(
                {
                    "system": run["label"],
                    "system_key": run["key"],
                    "dataset": "FLEURS Hindi",
                    "split": data["split"],
                    "normalization": mode,
                    "samples": data["num_samples"],
                    "audio_minutes": data["total_audio_seconds"] / 60.0,
                    "decode_minutes": data["total_decode_seconds"] / 60.0,
                    "rtf": data["rtf"],
                    "wer": metrics["wer"],
                    "cer": metrics["cer"],
                    "substitutions": metrics["substitutions"],
                    "deletions": metrics["deletions"],
                    "insertions": metrics["insertions"],
                    "hits": metrics["hits"],
                    "reference_words": metrics["num_reference_words"],
                    "english_reference_words": metrics["english_reference_words"],
                    "number_reference_words": metrics["number_reference_words"],
                    "code_mixed_samples": metrics["code_mixed_samples"],
                    "exact_match": metrics["exact_match"],
                    "checkpoint": data["checkpoint"],
                }
            )
    return rows


def save_metric_charts(df: pd.DataFrame) -> dict[str, Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    chart_paths: dict[str, Path] = {}

    metric_df = df.melt(
        id_vars=["system", "normalization"],
        value_vars=["wer", "cer"],
        var_name="metric",
        value_name="value",
    )
    fig, ax = plt.subplots(figsize=(10, 5.5))
    labels = []
    values = []
    colors = []
    palette = {
        ("whisper", "wer"): "#2f6f9f",
        ("whisper", "cer"): "#7aa95c",
        ("indic", "wer"): "#b66d3c",
        ("indic", "cer"): "#7c5aa6",
    }
    for _, row in metric_df.iterrows():
        labels.append(f"{row['system']}\n{row['normalization']} {row['metric'].upper()}")
        values.append(row["value"])
        colors.append(palette[(row["normalization"], row["metric"])])
    ax.bar(range(len(values)), values, color=colors)
    ax.set_ylabel("Error rate")
    ax.set_title("WER/CER Comparison on FLEURS Hindi Test")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.25)
    for i, value in enumerate(values):
        ax.text(i, value + 0.02, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    chart_paths["wer_cer"] = FIG_DIR / "team_wer_cer_comparison.png"
    fig.savefig(chart_paths["wer_cer"], dpi=180)
    plt.close(fig)

    indic = df[df["normalization"] == "indic"].copy()
    error_cols = ["substitutions", "deletions", "insertions"]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = range(len(indic))
    bottom = [0] * len(indic)
    colors = ["#4f7cac", "#c26b5d", "#d3a43f"]
    for col, color in zip(error_cols, colors):
        vals = indic[col].tolist()
        ax.bar(x, vals, bottom=bottom, label=col.title(), color=color)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_xticks(list(x))
    ax.set_xticklabels(indic["system"], rotation=15, ha="right")
    ax.set_ylabel("Error count")
    ax.set_title("Indic-Normalized Error Breakdown")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    chart_paths["errors"] = FIG_DIR / "team_error_breakdown_indic.png"
    fig.savefig(chart_paths["errors"], dpi=180)
    plt.close(fig)

    runtime = indic[["system", "rtf"]].copy()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(runtime["system"], runtime["rtf"], color=["#2f6f9f", "#b66d3c"])
    ax.set_ylabel("Real-time factor")
    ax.set_title("Inference Speed, Lower is Faster")
    ax.grid(axis="y", alpha=0.25)
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    chart_paths["rtf"] = FIG_DIR / "team_runtime_rtf.png"
    fig.savefig(chart_paths["rtf"], dpi=180)
    plt.close(fig)

    return chart_paths


def pct_delta(after: float, before: float) -> float:
    return (after - before) / before * 100.0


def write_reports(df: pd.DataFrame, charts: dict[str, Path]) -> tuple[Path, Path, Path]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary_csv = REPORT_DIR / "team_results_summary.csv"
    md_path = REPORT_DIR / "team_results_report.md"
    html_path = REPORT_DIR / "team_results_report.html"

    df_out = df[
        [
            "system",
            "dataset",
            "split",
            "normalization",
            "samples",
            "audio_minutes",
            "rtf",
            "wer",
            "cer",
            "substitutions",
            "deletions",
            "insertions",
            "checkpoint",
        ]
    ].copy()
    df_out["audio_minutes"] = df_out["audio_minutes"].round(2)
    df_out["rtf"] = df_out["rtf"].round(3)
    df_out["wer"] = df_out["wer"].round(4)
    df_out["cer"] = df_out["cer"].round(4)
    df_out.to_csv(summary_csv, index=False)

    base_indic = df[(df["system_key"] == "baseline") & (df["normalization"] == "indic")].iloc[0]
    ft_indic = df[(df["system_key"] == "fine_tuned") & (df["normalization"] == "indic")].iloc[0]
    base_whisper = df[(df["system_key"] == "baseline") & (df["normalization"] == "whisper")].iloc[0]
    ft_whisper = df[(df["system_key"] == "fine_tuned") & (df["normalization"] == "whisper")].iloc[0]

    wer_delta = pct_delta(ft_indic["wer"], base_indic["wer"])
    cer_delta = pct_delta(ft_indic["cer"], base_indic["cer"])
    whisper_wer_delta = pct_delta(ft_whisper["wer"], base_whisper["wer"])
    whisper_cer_delta = pct_delta(ft_whisper["cer"], base_whisper["cer"])

    md = f"""# Hindi ASR Evaluation Report

## Executive Summary

This report summarizes the completed Hindi ASR evaluation on the FLEURS Hindi test subset. The best-performing system in this run is the pretrained Whisper Base baseline.

Fine-tuning with LoRA on the available FLEURS Hindi training subset did not improve recognition accuracy. Under Indic normalization, WER increased from {base_indic['wer']:.4f} to {ft_indic['wer']:.4f}, and CER increased from {base_indic['cer']:.4f} to {ft_indic['cer']:.4f}. The recommended model from this experiment is therefore the pretrained Whisper Base baseline.

## Experiment Setup

| Item | Value |
|---|---:|
| Dataset evaluated | FLEURS Hindi |
| Evaluation split | test |
| Evaluation samples | {int(base_indic['samples'])} |
| Audio evaluated | {base_indic['audio_minutes']:.2f} minutes |
| Hardware | NVIDIA GeForce GTX 1650, 4 GB VRAM |
| Baseline model | openai/whisper-base |
| Fine-tuning method | LoRA |
| Fine-tuning samples | 510 |
| Dev samples | 128 |
| Epochs | 1 |
| Precision | fp16 |

## Main Results

| System | Normalization | WER | CER | RTF |
|---|---|---:|---:|---:|
| Whisper Base Baseline | Whisper-style | {base_whisper['wer']:.4f} | {base_whisper['cer']:.4f} | {base_whisper['rtf']:.3f} |
| Whisper Base LoRA Fine-Tuned | Whisper-style | {ft_whisper['wer']:.4f} | {ft_whisper['cer']:.4f} | {ft_whisper['rtf']:.3f} |
| Whisper Base Baseline | Indic | {base_indic['wer']:.4f} | {base_indic['cer']:.4f} | {base_indic['rtf']:.3f} |
| Whisper Base LoRA Fine-Tuned | Indic | {ft_indic['wer']:.4f} | {ft_indic['cer']:.4f} | {ft_indic['rtf']:.3f} |

## Change After Fine-Tuning

| Metric | Whisper-style delta | Indic delta |
|---|---:|---:|
| WER | {whisper_wer_delta:+.1f}% | {wer_delta:+.1f}% |
| CER | {whisper_cer_delta:+.1f}% | {cer_delta:+.1f}% |

Positive deltas mean higher error, so the fine-tuned model is worse in this run.

## Visualizations

### WER/CER Comparison

![WER/CER comparison](figures/{charts['wer_cer'].name})

### Indic-Normalized Error Breakdown

![Indic error breakdown](figures/{charts['errors'].name})

### Inference Speed

![Runtime RTF](figures/{charts['rtf'].name})

## Error Breakdown, Indic Normalization

| System | Substitutions | Deletions | Insertions | Reference words |
|---|---:|---:|---:|---:|
| Whisper Base Baseline | {int(base_indic['substitutions'])} | {int(base_indic['deletions'])} | {int(base_indic['insertions'])} | {int(base_indic['reference_words'])} |
| Whisper Base LoRA Fine-Tuned | {int(ft_indic['substitutions'])} | {int(ft_indic['deletions'])} | {int(ft_indic['insertions'])} | {int(ft_indic['reference_words'])} |

The fine-tuned model produced many more insertions, which is the main reason its WER increased.

## Recommendation

Use the pretrained Whisper Base baseline for the current Hindi ASR prototype. Do not use the LoRA fine-tuned checkpoint from this run for deployment because it degraded both WER and CER on the completed FLEURS Hindi evaluation.

Next experiment: train with more in-domain Hindi/Hinglish data and use a larger validation set before selecting a checkpoint.
"""

    md_path.write_text(md, encoding="utf-8")

    html_table = df_out.to_html(index=False, escape=False)
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Hindi ASR Evaluation Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #222; line-height: 1.45; }}
    h1, h2 {{ color: #17324d; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px; font-size: 14px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background: #eef3f7; }}
    img {{ max-width: 920px; width: 100%; border: 1px solid #ddd; margin: 10px 0 28px; }}
    .callout {{ background: #f6f8fa; border-left: 4px solid #2f6f9f; padding: 12px 16px; }}
  </style>
</head>
<body>
  <h1>Hindi ASR Evaluation Report</h1>
  <div class="callout">
    <strong>Recommendation:</strong> Use the pretrained Whisper Base baseline for the current prototype.
    The LoRA fine-tuned model degraded WER/CER on the completed FLEURS Hindi evaluation.
  </div>
  <h2>Main Results</h2>
  {html_table}
  <h2>WER/CER Comparison</h2>
  <img src="figures/{charts['wer_cer'].name}" alt="WER/CER comparison">
  <h2>Indic-Normalized Error Breakdown</h2>
  <img src="figures/{charts['errors'].name}" alt="Indic error breakdown">
  <h2>Inference Speed</h2>
  <img src="figures/{charts['rtf'].name}" alt="Runtime RTF">
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")
    return summary_csv, md_path, html_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a team-facing report from two ASR metric JSON files.")
    parser.add_argument(
        "--baseline-metrics",
        default=str(METRICS_DIR / "whisper_base_baseline_gpu_test.json"),
        help="Metric JSON for the baseline ASR system.",
    )
    parser.add_argument(
        "--fine-tuned-metrics",
        default=str(METRICS_DIR / "fine_tuned_gpu_final_test.json"),
        help="Metric JSON for the fine-tuned ASR system.",
    )
    parser.add_argument("--baseline-label", default="Whisper Base Baseline")
    parser.add_argument("--fine-tuned-label", default="Whisper Base LoRA Fine-Tuned")
    args = parser.parse_args()

    runs = [
        {"label": args.baseline_label, "key": "baseline", "path": Path(args.baseline_metrics)},
        {"label": args.fine_tuned_label, "key": "fine_tuned", "path": Path(args.fine_tuned_metrics)},
    ]
    rows = load_metrics(runs)
    df = pd.DataFrame(rows)
    charts = save_metric_charts(df)
    summary_csv, md_path, html_path = write_reports(df, charts)
    print(f"Summary CSV: {summary_csv}")
    print(f"Markdown report: {md_path}")
    print(f"HTML report: {html_path}")
    print("Figures:")
    for path in charts.values():
        print(f"  {path}")


if __name__ == "__main__":
    sys.exit(main())
