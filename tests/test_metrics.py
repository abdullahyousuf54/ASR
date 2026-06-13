import pytest

pytest.importorskip("jiwer")

from asr_pipeline.metrics import compute_asr_metrics


def test_metrics_exact_match():
    result = compute_asr_metrics(["hello world"], ["hello world"], mode="whisper")
    assert result.wer == 0
    assert result.cer == 0
    assert result.exact_match == 1


def test_metrics_detects_errors():
    result = compute_asr_metrics(["hello world"], ["hello"], mode="whisper")
    assert result.wer > 0
    assert result.deletions == 1
