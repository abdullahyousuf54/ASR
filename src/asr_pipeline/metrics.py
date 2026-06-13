from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import jiwer

from .normalization import is_english_token, normalize_for_mode


_NUMBER_RE = re.compile(r"^\d+(?:[.,:/-]\d+)*$")


@dataclass
class MetricResult:
    wer: float
    cer: float
    substitutions: int
    deletions: int
    insertions: int
    hits: int
    num_reference_words: int
    english_reference_words: int
    number_reference_words: int
    code_mixed_samples: int
    exact_match: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "wer": self.wer,
            "cer": self.cer,
            "substitutions": self.substitutions,
            "deletions": self.deletions,
            "insertions": self.insertions,
            "hits": self.hits,
            "num_reference_words": self.num_reference_words,
            "english_reference_words": self.english_reference_words,
            "number_reference_words": self.number_reference_words,
            "code_mixed_samples": self.code_mixed_samples,
            "exact_match": self.exact_match,
        }


def _safe_list(values: Iterable[str]) -> list[str]:
    return ["" if v is None else str(v) for v in values]


def compute_asr_metrics(
    references: Iterable[str],
    predictions: Iterable[str],
    mode: str = "indic",
) -> MetricResult:
    refs = [normalize_for_mode(x, mode) for x in _safe_list(references)]
    hyps = [normalize_for_mode(x, mode) for x in _safe_list(predictions)]

    if len(refs) != len(hyps):
        raise ValueError("references and predictions must have the same length")

    if not refs:
        return MetricResult(0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0)

    word_output = jiwer.process_words(refs, hyps)
    exact = sum(1 for r, h in zip(refs, hyps) if r == h) / len(refs)
    ref_tokens = [tok for ref in refs for tok in ref.split()]
    english_words = sum(1 for tok in ref_tokens if is_english_token(tok))
    number_words = sum(1 for tok in ref_tokens if _NUMBER_RE.match(tok))
    code_mixed = sum(
        1
        for ref in refs
        if any(is_english_token(tok) for tok in ref.split()) and re.search(r"[\u0900-\u097f]", ref)
    )

    return MetricResult(
        wer=float(word_output.wer),
        cer=float(jiwer.cer(refs, hyps)),
        substitutions=int(word_output.substitutions),
        deletions=int(word_output.deletions),
        insertions=int(word_output.insertions),
        hits=int(word_output.hits),
        num_reference_words=len(ref_tokens),
        english_reference_words=english_words,
        number_reference_words=number_words,
        code_mixed_samples=code_mixed,
        exact_match=exact,
    )


def group_metrics(rows: list[dict], mode: str = "indic") -> dict[str, dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        key = row.get("dataset", "unknown")
        grouped.setdefault(key, []).append(row)

    result = {}
    for dataset, ds_rows in grouped.items():
        result[dataset] = compute_asr_metrics(
            [r.get("reference", "") for r in ds_rows],
            [r.get("prediction", "") for r in ds_rows],
            mode=mode,
        ).to_dict()
    result["overall"] = compute_asr_metrics(
        [r.get("reference", "") for r in rows],
        [r.get("prediction", "") for r in rows],
        mode=mode,
    ).to_dict()
    return result


def error_category_counts(references: Iterable[str]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for ref in _safe_list(references):
        tokens = ref.split()
        if any(is_english_token(tok) for tok in tokens):
            counts["english_words"] += 1
        if any(_NUMBER_RE.match(tok) for tok in tokens):
            counts["numbers"] += 1
        if any(tok[:1].isupper() and is_english_token(tok) for tok in tokens):
            counts["latin_proper_noun_candidates"] += 1
    return dict(counts)

