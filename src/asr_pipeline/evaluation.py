from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from datasets import Dataset, DatasetDict, load_from_disk
from tqdm.auto import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline

from .audio import audio_duration_seconds, load_audio_mono_16k
from .metrics import compute_asr_metrics, group_metrics
from .utils import current_gpu_memory_mb, get_logger, write_json


LOGGER = get_logger(__name__)


def _load_model_and_processor(cfg: dict[str, Any], checkpoint: str | None = None):
    model_name = checkpoint or cfg["model"]["whisper_name"]
    checkpoint_path = Path(checkpoint) if checkpoint else None
    is_lora = bool(checkpoint_path and (checkpoint_path / "adapter_config.json").exists())
    processor_source = model_name
    has_processor = bool(
        checkpoint_path
        and (checkpoint_path / "preprocessor_config.json").exists()
        and ((checkpoint_path / "tokenizer.json").exists() or (checkpoint_path / "vocab.json").exists())
    )
    if is_lora or (checkpoint_path and not has_processor):
        processor_source = cfg["model"]["whisper_name"]

    try:
        processor = WhisperProcessor.from_pretrained(
            processor_source,
            language=cfg["model"]["language"],
            task=cfg["model"]["task"],
        )
        if is_lora:
            from peft import PeftModel

            model = WhisperForConditionalGeneration.from_pretrained(cfg["model"]["whisper_name"])
            model = PeftModel.from_pretrained(model, checkpoint)
            if hasattr(model, "merge_and_unload"):
                model = model.merge_and_unload()
        else:
            model = WhisperForConditionalGeneration.from_pretrained(model_name)
    except RuntimeError as exc:
        fallback = cfg["model"].get("fallback_whisper_name")
        if checkpoint is None and fallback and cfg["model"].get("use_fallback_if_oom", True) and "out of memory" in str(exc).lower():
            LOGGER.warning("Falling back to %s after OOM while loading %s", fallback, cfg["model"]["whisper_name"])
            processor = WhisperProcessor.from_pretrained(fallback, language=cfg["model"]["language"], task=cfg["model"]["task"])
            model = WhisperForConditionalGeneration.from_pretrained(fallback)
        else:
            raise
    model.generation_config.language = cfg["model"]["language"]
    model.generation_config.task = cfg["model"]["task"]
    model.generation_config.forced_decoder_ids = None
    return model, processor


def _asr_pipeline(cfg: dict[str, Any], checkpoint: str | None = None):
    model, processor = _load_model_and_processor(cfg, checkpoint)
    device = 0 if torch.cuda.is_available() else -1
    dtype = torch.float16 if torch.cuda.is_available() and cfg["training"].get("fp16", True) else torch.float32
    if torch.cuda.is_available():
        model = model.to(dtype=dtype)
    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=device,
        torch_dtype=dtype,
    )


def _audio_input(example: dict, sampling_rate: int) -> dict:
    audio = example["audio"]
    if isinstance(audio, str):
        start = float(example.get("start_seconds") or 0.0)
        end = example.get("end_seconds")
        duration = float(end) - start if end is not None else None
        return load_audio_mono_16k(audio, sampling_rate, offset=start, duration=duration)
    return audio


def evaluate_model(
    cfg: dict[str, Any],
    checkpoint: str | None,
    run_name: str,
    split: str = "test",
) -> dict[str, Any]:
    dataset_dict: DatasetDict = load_from_disk(cfg["paths"]["processed_dir"])
    if split not in dataset_dict:
        raise ValueError(f"Split {split} not found in {cfg['paths']['processed_dir']}")

    ds: Dataset = dataset_dict[split]
    max_samples = cfg["evaluation"].get("max_eval_samples")
    if max_samples:
        ds = ds.select(range(min(int(max_samples), len(ds))))

    asr = _asr_pipeline(cfg, checkpoint)
    generation_cfg = cfg["model"].get("generation", {})
    generate_kwargs = {
        "language": cfg["model"]["language"],
        "task": cfg["model"]["task"],
        "max_new_tokens": int(generation_cfg.get("max_new_tokens", 225)),
        "num_beams": int(generation_cfg.get("num_beams", 1)),
    }

    rows = []
    total_audio_seconds = 0.0
    total_decode_seconds = 0.0
    sampling_rate = int(cfg["audio"].get("sampling_rate", 16000))

    for example in tqdm(ds, desc=f"evaluate {run_name}:{split}"):
        audio = _audio_input(example, sampling_rate)
        duration = audio_duration_seconds(audio)
        start = time.perf_counter()
        output = asr(audio, generate_kwargs=generate_kwargs)
        elapsed = time.perf_counter() - start
        total_audio_seconds += duration
        total_decode_seconds += elapsed
        rows.append(
            {
                "uid": example.get("uid"),
                "dataset": example.get("dataset"),
                "split": split,
                "speaker_id": example.get("speaker_id"),
                "reference": example.get("transcript_indic") or example.get("transcript_raw") or "",
                "reference_raw": example.get("transcript_raw") or "",
                "prediction": output.get("text", "").strip(),
                "duration_seconds": duration,
                "decode_seconds": elapsed,
                "run_name": run_name,
                "checkpoint": checkpoint or cfg["model"]["whisper_name"],
            }
        )

    pred_dir = Path(cfg["paths"]["predictions_dir"])
    metrics_dir = Path(cfg["paths"]["metrics_dir"])
    pred_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    pred_csv = pred_dir / f"{run_name}_{split}.csv"
    pred_jsonl = pred_dir / f"{run_name}_{split}.jsonl"
    pd.DataFrame(rows).to_csv(pred_csv, index=False)
    with pred_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics = {
        "run_name": run_name,
        "split": split,
        "checkpoint": checkpoint or cfg["model"]["whisper_name"],
        "num_samples": len(rows),
        "total_audio_seconds": total_audio_seconds,
        "total_decode_seconds": total_decode_seconds,
        "rtf": total_decode_seconds / total_audio_seconds if total_audio_seconds else None,
        "max_gpu_memory_mb": current_gpu_memory_mb(),
        "modes": {},
    }
    for mode in cfg["text"].get("evaluation_modes", ["whisper", "indic"]):
        metrics["modes"][mode] = compute_asr_metrics(
            [r["reference"] for r in rows],
            [r["prediction"] for r in rows],
            mode=mode,
        ).to_dict()
        metrics.setdefault("by_dataset", {})[mode] = group_metrics(rows, mode=mode)

    metric_path = metrics_dir / f"{run_name}_{split}.json"
    write_json(metric_path, metrics)
    LOGGER.info("Saved predictions to %s and metrics to %s", pred_csv, metric_path)
    return {"predictions_csv": str(pred_csv), "metrics_json": str(metric_path), "metrics": metrics}
