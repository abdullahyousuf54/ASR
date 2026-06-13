from __future__ import annotations

import re
import unicodedata
from functools import lru_cache


_WHITESPACE_RE = re.compile(r"\s+")
_BRACKET_NOISE_RE = re.compile(r"\[(?:noise|music|silence|laughter|applause|unk|inaudible)[^\]]*\]", re.I)
_ANGLE_NOISE_RE = re.compile(r"<(?:noise|music|silence|unk|inaudible)[^>]*>", re.I)
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([।,.!?;:%])")
_MULTI_PUNCT_RE = re.compile(r"([।,.!?]){2,}")


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text or "")


def light_cleanup(text: str) -> str:
    text = normalize_unicode(text)
    text = _BRACKET_NOISE_RE.sub(" ", text)
    text = _ANGLE_NOISE_RE.sub(" ", text)
    text = text.replace("\u200c", "").replace("\u200d", "")
    text = _MULTI_PUNCT_RE.sub(lambda m: m.group(1), text)
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


@lru_cache(maxsize=1)
def _hindi_normalizer():
    try:
        from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

        factory = IndicNormalizerFactory()
        return factory.get_normalizer("hi")
    except Exception:
        return None


def indic_normalize(text: str) -> str:
    """Hindi-faithful normalization used for training labels and Indic metrics."""
    text = light_cleanup(text)
    normalizer = _hindi_normalizer()
    if normalizer is not None:
        text = normalizer.normalize(text)
    text = text.replace(" ।", "।")
    return _WHITESPACE_RE.sub(" ", text).strip()


@lru_cache(maxsize=1)
def _whisper_normalizer():
    try:
        from transformers.models.whisper.english_normalizer import BasicTextNormalizer

        return BasicTextNormalizer(remove_diacritics=True, split_letters=False)
    except Exception:
        return None


def whisper_normalize(text: str) -> str:
    """Aggressive Whisper-style normalization for comparable WER reporting."""
    text = light_cleanup(text).lower()
    normalizer = _whisper_normalizer()
    if normalizer is not None:
        text = normalizer(text)
    text = re.sub(r"[^\w\s\u0900-\u097f]", " ", text, flags=re.UNICODE)
    return _WHITESPACE_RE.sub(" ", text).strip()


def normalize_for_mode(text: str, mode: str) -> str:
    if mode == "indic":
        return indic_normalize(text)
    if mode == "whisper":
        return whisper_normalize(text)
    if mode == "raw":
        return light_cleanup(text)
    raise ValueError(f"Unsupported normalization mode: {mode}")


def is_english_token(token: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z'\-]*", token or ""))


def contains_devanagari(text: str) -> bool:
    return bool(re.search(r"[\u0900-\u097f]", text or ""))

