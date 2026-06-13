from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
from datasets import load_from_disk
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asr_pipeline.audio import audio_duration_seconds, load_audio_mono_16k
from asr_pipeline.config import ensure_project_dirs, load_config
from asr_pipeline.metrics import compute_asr_metrics, group_metrics
from asr_pipeline.utils import current_gpu_memory_mb, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Optional Indic-native ASR baseline via HF pipeline.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--run-name", default="indic_native_baseline")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_project_dirs(cfg)
    model_name = args.model_name or cfg["evaluation"]["optional_indic_baseline"]["model_name"]

    try:
        import torch
        from transformers import pipeline

        device = 0 if torch.cuda.is_available() else -1
        asr = pipeline("automatic-speech-recognition", model=model_name, trust_remote_code=True, device=device)
    except Exception as exc:
        print(f"Could not load optional Indic baseline {model_name}: {exc}")
        return

    dsd = load_from_disk(cfg["paths"]["processed_dir"])
    ds = dsd[args.split]
    max_samples = cfg["evaluation"].get("max_eval_samples")
    if max_samples:
        ds = ds.select(range(min(int(max_samples), len(ds))))

    rows = []
    total_audio = 0.0
    total_decode = 0.0
    sampling_rate = int(cfg["audio"].get("sampling_rate", 16000))
    for example in tqdm(ds, desc=f"evaluate {args.run_name}:{args.split}"):
        audio = example["audio"]
        if isinstance(audio, str):
            start_seconds = float(example.get("start_seconds") or 0.0)
            end_seconds = example.get("end_seconds")
            duration = float(end_seconds) - start_seconds if end_seconds is not None else None
            audio = load_audio_mono_16k(audio, sampling_rate, offset=start_seconds, duration=duration)
        duration = audio_duration_seconds(audio)
        start = time.perf_counter()
        output = asr(audio)
        elapsed = time.perf_counter() - start
        total_audio += duration
        total_decode += elapsed
        rows.append(
            {
                "uid": example.get("uid"),
                "dataset": example.get("dataset"),
                "split": args.split,
                "speaker_id": example.get("speaker_id"),
                "reference": example.get("transcript_indic") or example.get("transcript_raw") or "",
                "reference_raw": example.get("transcript_raw") or "",
                "prediction": output.get("text", "").strip(),
                "duration_seconds": duration,
                "decode_seconds": elapsed,
                "run_name": args.run_name,
                "checkpoint": model_name,
            }
        )

    pred_dir = Path(cfg["paths"]["predictions_dir"])
    metrics_dir = Path(cfg["paths"]["metrics_dir"])
    pred_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    pred_csv = pred_dir / f"{args.run_name}_{args.split}.csv"
    pd.DataFrame(rows).to_csv(pred_csv, index=False)
    with (pred_dir / f"{args.run_name}_{args.split}.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics = {
        "run_name": args.run_name,
        "split": args.split,
        "checkpoint": model_name,
        "num_samples": len(rows),
        "rtf": total_decode / total_audio if total_audio else None,
        "max_gpu_memory_mb": current_gpu_memory_mb(),
        "modes": {},
        "by_dataset": {},
    }
    for mode in cfg["text"].get("evaluation_modes", ["whisper", "indic"]):
        metrics["modes"][mode] = compute_asr_metrics(
            [r["reference"] for r in rows],
            [r["prediction"] for r in rows],
            mode=mode,
        ).to_dict()
        metrics["by_dataset"][mode] = group_metrics(rows, mode=mode)
    metric_path = metrics_dir / f"{args.run_name}_{args.split}.json"
    write_json(metric_path, metrics)
    print(metrics)


if __name__ == "__main__":
    main()
