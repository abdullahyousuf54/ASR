from __future__ import annotations

import os
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tqdm.auto import tqdm

from .metrics import compute_asr_metrics
from .utils import get_logger, write_json


LOGGER = get_logger(__name__)


PROMPT_TEMPLATE = """You are correcting Hindi/Hinglish ASR output.

Rules:
- Do not paraphrase.
- Do not add information.
- Do not reorder the sentence unless required to fix an obvious ASR error.
- Preserve named entities as much as possible.
- Keep Hindi in Hindi script and English words in English script.
- Correct only spelling, script, punctuation, number formatting, and obvious ASR mistakes.
- If uncertain, return the input unchanged.
- Output only the corrected text.

ASR output:
{text}
"""


def _edit_ratio(original: str, corrected: str) -> float:
    if not original and corrected:
        return 1.0
    return 1.0 - SequenceMatcher(None, original, corrected).ratio()


class CorrectionClient:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.llm_cfg = cfg.get("llm", {})
        self.provider = self.llm_cfg.get("provider", "none")

    def correct(self, text: str) -> str:
        if self.provider in {None, "", "none"}:
            return text
        prompt = PROMPT_TEMPLATE.format(text=text)
        if self.provider == "local_http":
            return self._local_http(prompt, text)
        if self.provider == "openai_compatible":
            return self._openai_compatible(prompt, text)
        raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def _guard(self, original: str, corrected: str) -> str:
        corrected = (corrected or "").strip().strip('"')
        if not corrected:
            return original
        max_ratio = float(self.llm_cfg.get("max_edit_ratio", 0.55))
        if _edit_ratio(original, corrected) > max_ratio:
            LOGGER.warning("LLM correction rejected by edit-ratio guard: %s -> %s", original, corrected)
            return original
        return corrected

    def _local_http(self, prompt: str, original: str) -> str:
        endpoint = self.llm_cfg.get("endpoint")
        if not endpoint:
            raise ValueError("llm.endpoint is required for local_http provider")
        payload = {
            "model": self.llm_cfg.get("model"),
            "prompt": prompt,
            "temperature": float(self.llm_cfg.get("temperature", 0.0)),
        }
        response = requests.post(endpoint, json=payload, timeout=int(self.llm_cfg.get("timeout_seconds", 60)))
        response.raise_for_status()
        data = response.json()
        corrected = data.get("text") or data.get("response") or data.get("output") or ""
        return self._guard(original, corrected)

    def _openai_compatible(self, prompt: str, original: str) -> str:
        from openai import OpenAI

        api_key_env = self.llm_cfg.get("api_key_env", "OPENAI_API_KEY")
        client = OpenAI(api_key=os.environ.get(api_key_env), base_url=self.llm_cfg.get("endpoint") or None)
        response = client.chat.completions.create(
            model=self.llm_cfg["model"],
            temperature=float(self.llm_cfg.get("temperature", 0.0)),
            messages=[{"role": "user", "content": prompt}],
        )
        corrected = response.choices[0].message.content or ""
        return self._guard(original, corrected)


def correct_predictions(cfg: dict[str, Any], predictions_path: str | Path) -> dict[str, Any]:
    path = Path(predictions_path)
    df = pd.read_csv(path)
    client = CorrectionClient(cfg)
    corrected = []
    for text in tqdm(df["prediction"].fillna("").astype(str).tolist(), desc="LLM correction"):
        corrected.append(client.correct(text))

    df["prediction_before_llm"] = df["prediction"]
    df["prediction"] = corrected
    run_name = f"{df['run_name'].iloc[0]}_llm" if "run_name" in df.columns and len(df) else "llm_corrected"
    df["run_name"] = run_name

    out_dir = Path(cfg["paths"]["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{path.stem}_llm.csv"
    df.to_csv(out_csv, index=False)

    metrics = {
        "run_name": run_name,
        "source_predictions": str(path),
        "corrected_predictions": str(out_csv),
        "modes": {},
    }
    for mode in cfg["text"].get("evaluation_modes", ["whisper", "indic"]):
        before = compute_asr_metrics(df["reference"].tolist(), df["prediction_before_llm"].tolist(), mode=mode).to_dict()
        after = compute_asr_metrics(df["reference"].tolist(), df["prediction"].tolist(), mode=mode).to_dict()
        metrics["modes"][mode] = {"before": before, "after": after, "delta_wer": after["wer"] - before["wer"]}

    metrics_path = Path(cfg["paths"]["metrics_dir"]) / f"{path.stem}_llm.json"
    write_json(metrics_path, metrics)
    return {"predictions_csv": str(out_csv), "metrics_json": str(metrics_path), "metrics": metrics}

