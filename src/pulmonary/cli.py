"""File-based command line workflow. Run from the repository root."""

import argparse
import json
import os
from pathlib import Path

from pulmonary.common import read_config, save_json


def download(config):
    # Keep downloads/caches in the project instead of relying on a Colab runtime.
    os.environ.setdefault("KAGGLEHUB_CACHE", str(Path("data/.cache").resolve()))
    import kagglehub

    path = kagglehub.dataset_download("raddar/chest-xrays-indiana-university")
    save_json(
        "data/download_manifest.json",
        {
            "source": "https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university",
            "downloaded_path": str(path),
            "instruction": "Set data_root in your config to downloaded_path, then run prepare.",
        },
    )
    import yaml

    try:
        relative = Path(path).resolve().relative_to(Path.cwd()).as_posix()
    except ValueError:
        relative = str(path)
    Path("configs/local.yaml").write_text(
        yaml.safe_dump({"extends": "default.yaml", "data_root": relative}), encoding="utf-8"
    )
    return {
        "downloaded_path": str(path),
        "next": "Use --config configs/local.yaml prepare",
    }


def report(config):
    from pulmonary.reporting import report as build_report

    return build_report(config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "command",
        choices=[
            "download",
            "prepare",
            "prepare-metadata",
            "train-text",
            "train-images",
            "train-fusion",
            "export",
            "evaluate-test",
            "explain-examples",
            "report",
        ],
    )
    args = parser.parse_args()
    config = read_config(args.config)
    if args.command == "download":
        result = download(config)
    elif args.command in {"prepare", "prepare-metadata"}:
        from pulmonary.data import prepare

        result = prepare(config, metadata_only=args.command == "prepare-metadata")
    elif args.command == "train-text":
        from pulmonary.text_models import train_text

        result = train_text(config)
    elif args.command == "train-images":
        from pulmonary.image_models import train_images

        result = train_images(config)
    elif args.command == "train-fusion":
        from pulmonary.fusion import train_fusion

        result = train_fusion(config)
    elif args.command == "export":
        from pulmonary.artifacts import export_bundle

        result = export_bundle(config)
    elif args.command == "evaluate-test":
        from pulmonary.artifacts import evaluate_test

        result = evaluate_test(config)
    elif args.command == "explain-examples":
        from pulmonary.artifacts import explanation_examples

        result = explanation_examples(config)
    else:
        result = report(config)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
