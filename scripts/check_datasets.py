from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset

from asr_pipeline.config import load_config


def check_hf(spec: dict) -> tuple[str, str]:
    kwargs = {"path": spec["path"], "trust_remote_code": bool(spec.get("trust_remote_code", False))}
    name = spec.get("name")
    try:
        configs = get_dataset_config_names(**kwargs)
        if name and name not in configs:
            return "not_ready", f"config `{name}` not found; available sample: {configs[:10]}"
        splits = get_dataset_split_names(spec["path"], name, trust_remote_code=kwargs["trust_remote_code"])
        missing = [split for split in spec.get("splits", {}).values() if split not in splits]
        if missing:
            return "not_ready", f"missing splits {missing}; available: {splits}"
        first_role, first_split = next(iter(spec.get("splits", {"train": "train"}).items()))
        ds = load_dataset(
            spec["path"],
            name,
            split=f"{first_split}[:1]",
            trust_remote_code=kwargs["trust_remote_code"],
        )
        return "ready", f"configs ok, splits ok, sample columns: {ds.column_names}"
    except Exception as exc:
        return "blocked", f"{type(exc).__name__}: {exc}"


def check_local(spec: dict) -> tuple[str, str]:
    import glob

    paths = sorted(glob.glob(spec.get("manifest_glob", "")))
    if paths:
        return "ready", f"found {len(paths)} manifest(s)"
    return "manual", f"no manifests at {spec.get('manifest_glob')}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Check dataset availability for the ASR pipeline.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    for spec in cfg.get("datasets", []):
        if spec.get("enabled", True) is False:
            status = "disabled"
            detail = spec.get("access_note", "disabled in config")
        elif spec.get("kind") == "hf":
            status, detail = check_hf(spec)
        elif spec.get("kind") == "local_manifest":
            status, detail = check_local(spec)
        else:
            status, detail = "unknown", f"unsupported kind {spec.get('kind')}"
        print(f"{spec.get('id')}: {status} - {detail}")


if __name__ == "__main__":
    main()

