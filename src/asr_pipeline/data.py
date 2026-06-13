from __future__ import annotations

import glob
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import Audio, Dataset, DatasetDict, concatenate_datasets, load_dataset
from tqdm.auto import tqdm

from .audio import audio_duration_seconds, file_duration_seconds
from .normalization import indic_normalize, light_cleanup
from .utils import env_token, get_logger


LOGGER = get_logger(__name__)

TEXT_CANDIDATES = ("transcript", "sentence", "text", "transcription", "normalized_text")
AUDIO_CANDIDATES = ("audio", "audio_path", "path", "file", "filepath")
SPEAKER_CANDIDATES = ("speaker_id", "client_id", "user_id", "speaker", "utt_spk")


def _pick_column(columns: list[str], configured: str | None, candidates: tuple[str, ...]) -> str | None:
    if configured and configured in columns:
        return configured
    for cand in candidates:
        if cand in columns:
            return cand
    return None


def _load_hf_split(spec: dict[str, Any], split_name: str) -> Dataset | None:
    kwargs = {
        "path": spec["path"],
        "name": spec.get("name"),
        "split": split_name,
        "trust_remote_code": bool(spec.get("trust_remote_code", False)),
    }
    token = env_token()
    if token:
        kwargs["token"] = token
    try:
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return load_dataset(**kwargs)
    except Exception as exc:
        LOGGER.warning("Skipping %s split %s: %s", spec.get("id"), split_name, exc)
        return None


def _load_local_manifests(spec: dict[str, Any]) -> DatasetDict:
    paths = sorted(glob.glob(spec.get("manifest_glob", "")))
    if not paths:
        LOGGER.warning("No local manifests found for %s at %s", spec.get("id"), spec.get("manifest_glob"))
        return DatasetDict()

    frames = []
    for path in paths:
        p = Path(path)
        if p.suffix.lower() == ".jsonl":
            frame = pd.read_json(p, lines=True)
        else:
            frame = pd.read_csv(p)
        frame["_manifest_dir"] = str(p.parent)
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    split_col = spec.get("split_column", "split")
    if split_col not in df.columns:
        df = _speaker_safe_split_frame(df, spec, split_col)

    result = DatasetDict()
    for split_name, split_df in df.groupby(split_col):
        result[str(split_name)] = Dataset.from_pandas(split_df.reset_index(drop=True), preserve_index=False)
    return result


def _speaker_safe_split_frame(df: pd.DataFrame, spec: dict[str, Any], split_col: str) -> pd.DataFrame:
    speaker_col = spec.get("speaker_column", "speaker_id")
    if speaker_col not in df.columns:
        df[split_col] = "train"
        return df

    rng = np.random.default_rng(1337)
    speakers = np.asarray(sorted(df[speaker_col].fillna("unknown").astype(str).unique()))
    rng.shuffle(speakers)
    n = len(speakers)
    if n < 3:
        df[split_col] = "train"
        return df

    train_end = max(1, int(0.9 * n))
    dev_end = max(train_end + 1, int(0.95 * n))
    train_speakers = set(speakers[:train_end])
    dev_speakers = set(speakers[train_end:dev_end])

    def assign(speaker: object) -> str:
        speaker = str(speaker)
        if speaker in train_speakers:
            return "train"
        if speaker in dev_speakers:
            return "dev"
        return "test"

    df[split_col] = df[speaker_col].map(assign)
    return df


def _standardize_split(
    ds: Dataset,
    spec: dict[str, Any],
    role: str,
    split_name: str,
    sampling_rate: int,
    min_seconds: float,
    max_seconds: float,
) -> Dataset:
    columns = list(ds.column_names)
    text_col = _pick_column(columns, spec.get("text_column"), TEXT_CANDIDATES)
    audio_col = _pick_column(columns, spec.get("audio_column"), AUDIO_CANDIDATES)
    speaker_col = _pick_column(columns, spec.get("speaker_column"), SPEAKER_CANDIDATES)

    if text_col is None or audio_col is None:
        raise ValueError(f"{spec.get('id')} missing text/audio columns. Columns: {columns}")

    if audio_col == "audio":
        try:
            ds = ds.cast_column(audio_col, Audio(sampling_rate=sampling_rate))
        except Exception as exc:
            LOGGER.warning("Could not cast %s audio to %s Hz: %s", spec.get("id"), sampling_rate, exc)

    def convert(example: dict, idx: int) -> dict:
        raw = light_cleanup(str(example.get(text_col) or ""))
        speaker = str(example.get(speaker_col) or "unknown") if speaker_col else "unknown"
        audio_value = example.get(audio_col)
        if isinstance(audio_value, str) and not Path(audio_value).is_absolute() and example.get("_manifest_dir"):
            audio_value = str(Path(example["_manifest_dir"]) / audio_value)
        start_seconds = _safe_float(example.get("start_seconds", example.get("start", 0.0)), default=0.0)
        end_value = example.get("end_seconds", example.get("end"))
        end_seconds = _safe_float(end_value, default=None) if end_value not in {None, ""} else None
        return {
            "uid": f"{spec.get('id')}-{role}-{idx}",
            "dataset": spec.get("id"),
            "split": role,
            "source_split": split_name,
            "speaker_id": speaker,
            "audio": audio_value,
            "start_seconds": start_seconds,
            "end_seconds": end_seconds,
            "transcript_raw": raw,
            "transcript_indic": indic_normalize(raw),
            "domain": spec.get("domain", spec.get("id")),
            "access_note": spec.get("access_note", ""),
        }

    keep_cols = [
        "uid",
        "dataset",
        "split",
        "source_split",
        "speaker_id",
        "audio",
        "start_seconds",
        "end_seconds",
        "transcript_raw",
        "transcript_indic",
        "domain",
        "access_note",
    ]
    ds = ds.map(convert, with_indices=True, remove_columns=columns, desc=f"standardize {spec.get('id')}:{role}")

    def valid(example: dict) -> bool:
        text = example.get("transcript_indic") or ""
        if not text.strip():
            return False
        audio = example.get("audio")
        try:
            if isinstance(audio, str):
                if not Path(audio).exists():
                    return False
                start = float(example.get("start_seconds") or 0.0)
                end = example.get("end_seconds")
                if end is not None:
                    duration = float(end) - start
                else:
                    file_duration = file_duration_seconds(audio)
                    duration = file_duration if file_duration is not None else max_seconds
                return min_seconds <= duration <= max_seconds
            duration = audio_duration_seconds(audio)
            return min_seconds <= duration <= max_seconds
        except Exception:
            return False

    ds = ds.filter(valid, desc=f"validate {spec.get('id')}:{role}")
    return ds.select_columns(keep_cols)


def _safe_float(value: object, default: float | None = 0.0) -> float | None:
    try:
        return float(value)
    except Exception:
        return default


def _drop_speaker_leakage(role_buckets: dict[str, list[Dataset]]) -> dict[str, list[Dataset]]:
    if not role_buckets.get("train"):
        return role_buckets
    train = concatenate_datasets(role_buckets["train"])
    train_pairs = {
        (row["dataset"], row["speaker_id"])
        for row in train.select_columns(["dataset", "speaker_id"])
        if row.get("speaker_id") and row.get("speaker_id") != "unknown"
    }
    if not train_pairs:
        return role_buckets

    for role in ("dev", "test"):
        cleaned = []
        for ds in role_buckets.get(role, []):
            before = len(ds)
            ds = ds.filter(
                lambda row: (row["dataset"], row["speaker_id"]) not in train_pairs
                or not row.get("speaker_id")
                or row.get("speaker_id") == "unknown",
                desc=f"drop speaker leakage:{role}",
            )
            if len(ds) != before:
                LOGGER.info("Dropped %d leaked-speaker samples from %s", before - len(ds), role)
            if len(ds):
                cleaned.append(ds)
        role_buckets[role] = cleaned
    return role_buckets


def prepare_datasets(cfg: dict[str, Any]) -> DatasetDict:
    audio_cfg = cfg["audio"]
    sampling_rate = int(audio_cfg.get("sampling_rate", 16000))
    min_seconds = float(audio_cfg.get("min_seconds", 0.2))
    max_seconds = float(audio_cfg.get("max_seconds", 30.0))
    role_buckets: dict[str, list[Dataset]] = {"train": [], "dev": [], "test": []}

    for spec in tqdm(cfg.get("datasets", []), desc="datasets"):
        if spec.get("enabled", True) is False:
            LOGGER.info("Dataset %s disabled in config", spec.get("id"))
            continue
        kind = spec.get("kind")
        if kind == "hf":
            for role, split_name in spec.get("splits", {}).items():
                if role == "train" and not spec.get("use_for_train", False):
                    continue
                if role in {"dev", "test"} and not spec.get("use_for_eval", False) and not spec.get("use_for_train", False):
                    continue
                ds = _load_hf_split(spec, split_name)
                if ds is None:
                    continue
                try:
                    limit = spec.get("max_samples", {}).get(role)
                    if limit:
                        ds = ds.select(range(min(int(limit), len(ds))))
                    std = _standardize_split(ds, spec, role, split_name, sampling_rate, min_seconds, max_seconds)
                    if len(std):
                        role_buckets.setdefault(role, []).append(std)
                except Exception as exc:
                    LOGGER.warning("Skipping %s %s after load: %s", spec.get("id"), role, exc)
        elif kind == "local_manifest":
            dsd = _load_local_manifests(spec)
            for split_name, ds in dsd.items():
                role = "dev" if split_name in {"valid", "validation"} else split_name
                if role not in role_buckets:
                    role = "train"
                try:
                    std = _standardize_split(ds, spec, role, split_name, sampling_rate, min_seconds, max_seconds)
                    if len(std):
                        role_buckets[role].append(std)
                except Exception as exc:
                    LOGGER.warning("Skipping local %s %s: %s", spec.get("id"), split_name, exc)
        else:
            LOGGER.warning("Unknown dataset kind for %s: %s", spec.get("id"), kind)

    role_buckets = _drop_speaker_leakage(role_buckets)
    result = DatasetDict()
    for role, datasets in role_buckets.items():
        if datasets:
            result[role] = concatenate_datasets(datasets)
            LOGGER.info("Prepared %s split with %d samples", role, len(result[role]))
        else:
            LOGGER.warning("Prepared %s split is empty", role)
    return result


def save_prepared_dataset(cfg: dict[str, Any], dataset_dict: DatasetDict) -> None:
    if not dataset_dict or not any(len(split) for split in dataset_dict.values()):
        raise RuntimeError(
            "No usable samples were prepared. Check Hugging Face login/network access, "
            "dataset terms acceptance, or local manifests under data/raw/."
        )
    if "train" not in dataset_dict or "dev" not in dataset_dict:
        raise RuntimeError(
            "Prepared data is missing train/dev splits. Enable an accessible training dataset "
            "or add local manifests before running fine-tuning."
        )
    out = Path(cfg["paths"]["processed_dir"])
    tmp_out = out.with_name(f"{out.name}.tmp")
    if tmp_out.exists():
        shutil.rmtree(tmp_out)
    tmp_out.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(tmp_out))
    if out.exists():
        shutil.rmtree(out)
    tmp_out.replace(out)
    LOGGER.info("Saved processed dataset to %s", out)
