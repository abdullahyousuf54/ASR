from __future__ import annotations

from pathlib import Path

import numpy as np


def load_audio_mono_16k(
    path: str | Path,
    sampling_rate: int = 16000,
    offset: float = 0.0,
    duration: float | None = None,
) -> dict:
    import librosa

    array, sr = librosa.load(str(path), sr=sampling_rate, mono=True, offset=offset, duration=duration)
    return {"array": np.asarray(array, dtype=np.float32), "sampling_rate": sampling_rate}


def audio_duration_seconds(audio: dict) -> float:
    array = audio.get("array")
    sr = int(audio.get("sampling_rate") or 16000)
    if array is None or sr <= 0:
        return 0.0
    return float(len(array) / sr)


def file_duration_seconds(path: str | Path) -> float | None:
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return float(info.frames / info.samplerate)
    except Exception:
        return None
