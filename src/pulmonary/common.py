"""Configuration, provenance, and reproducibility utilities."""

import hashlib
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import yaml


def read_config(path="configs/default.yaml"):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if "extends" in config:
        parent = Path(path).parent / config.pop("extends")
        config = {**read_config(parent), **config}
    for key in ("output_root", "reports_root"):
        Path(config[key]).mkdir(parents=True, exist_ok=True)
    return config


def split_digest(config):
    return file_hash(Path(config["output_root"]) / "splits/manifest.json")


def require_verified_images(config):
    manifest = json.loads((Path(config["output_root"]) / "splits/manifest.json").read_text())
    if manifest.get("image_validation") != "complete":
        raise ValueError(
            "Images are not verified. Run prepare with the full dataset; metadata-only splits cannot train or serve image/fusion models."
        )


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def revision():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unversioned"


def source_digest():
    digest = hashlib.sha256()
    paths = (
        [Path("pyproject.toml")]
        + sorted(Path("src").rglob("*.py"))
        + sorted(Path("app").rglob("*.py"))
    )
    for path in paths:
        if path.exists():
            digest.update(path.as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def seed_everything(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def device_for(config):
    import torch

    setting = config.get("device", "auto")
    return (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if setting == "auto"
        else torch.device(setting)
    )


def load_splits(config, include_test=False):
    import pandas as pd

    root = Path(config["output_root"]) / "splits"
    manifest = json.loads((root / "manifest.json").read_text())
    names = ["train", "validation"] + (["test"] if include_test else [])
    frames = {
        name: pd.read_csv(
            root / f"{name}.csv", dtype={"uid": str, "group_id": str}, keep_default_na=False
        )
        for name in names
    }
    seen = set()
    for name, frame in frames.items():
        if file_hash(root / f"{name}.csv") != manifest["splits"][name]["sha256"]:
            raise ValueError(f"{name} split changed since preparation")
        groups = set(frame.group_id)
        if seen & groups:
            raise ValueError(f"Group overlap in {name}")
        seen.update(groups)
        if set(frame.target) != {0, 1}:
            raise ValueError(f"{name} must contain both classes")
    return frames
