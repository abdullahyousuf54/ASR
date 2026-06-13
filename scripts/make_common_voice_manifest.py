from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_manifest(root: Path, out_path: Path, include_other_as_train: bool = False) -> pd.DataFrame:
    duration_path = root / "clip_durations.tsv"
    durations = pd.read_csv(duration_path, sep="\t")
    duration_map = dict(zip(durations["clip"], durations["duration[ms]"]))

    split_files = [("train", "train.tsv"), ("dev", "dev.tsv"), ("test", "test.tsv")]
    if include_other_as_train:
        split_files.append(("train", "other.tsv"))

    frames = []
    for split, filename in split_files:
        path = root / filename
        if not path.exists():
            continue
        df = pd.read_csv(path, sep="\t")
        if "path" not in df.columns or "sentence" not in df.columns:
            raise ValueError(f"{path} must contain path and sentence columns")
        speaker = df["client_id"] if "client_id" in df.columns else "unknown"
        frame = pd.DataFrame(
            {
                "audio_path": df["path"].map(lambda p: str(Path("..") / "hi" / "clips" / str(p))),
                "transcript": df["sentence"].fillna(""),
                "speaker_id": speaker.fillna("unknown"),
                "split": split,
                "domain": "common_voice_official",
                "language_mix": "hindi",
                "duration_ms": df["path"].map(duration_map),
            }
        )
        frame = frame[frame["transcript"].astype(str).str.strip().ne("")]
        frame = frame[frame["duration_ms"].notna()]
        frames.append(frame)

    if not frames:
        raise RuntimeError(f"No Common Voice split TSV files found in {root}")

    manifest = pd.concat(frames, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_path, index=False, encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a local manifest for official Common Voice Hindi.")
    parser.add_argument("--root", default="data/raw/common_voice_official_hi/hi")
    parser.add_argument("--out", default="data/raw/common_voice_official_hi/manifests/common_voice_official_hi.csv")
    parser.add_argument("--include-other-as-train", action="store_true")
    args = parser.parse_args()

    manifest = build_manifest(Path(args.root), Path(args.out), args.include_other_as_train)
    print(f"Wrote {len(manifest)} rows to {args.out}")
    print(manifest.groupby("split")["duration_ms"].agg(rows="count", hours=lambda x: round(x.sum() / 3_600_000, 2)))


if __name__ == "__main__":
    main()

