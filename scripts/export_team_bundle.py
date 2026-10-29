"""Export the completed team reproduction without replacing the v1 bundle."""

import argparse
import importlib.metadata
import json
import shutil
from pathlib import Path

import joblib

from pulmonary.common import file_hash, save_json
from pulmonary.team_features import PREPROCESS, TeamTextModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="artifacts/team_classification")
    parser.add_argument("--output", default="artifacts/team_bundle")
    parser.add_argument(
        "--model-version",
        help="Override the version name, e.g. for a retrained model on the same cohort",
    )
    args = parser.parse_args()
    source, output = Path(args.source), Path(args.output)
    selection = json.loads((source / "selection.json").read_text())
    if not (source / "ablation_results.csv").is_file():
        raise ValueError("Complete held-out evaluation before exporting")
    if file_hash(source / "cohort_manifest.json") != selection["cohort_manifest"]:
        raise ValueError("Selection/cohort mismatch")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise FileExistsError("Use a new output directory for a different model version")
    for name in ["resnet50.pt", "densenet121.pt"]:
        shutil.copyfile(source / name, output / name)
    text = joblib.load(source / "text_model.joblib")
    # Older in-process training can retain the pre-refactor class path.
    text.__class__ = TeamTextModel
    joblib.dump(text, output / "text_model.joblib")
    files = {
        name: file_hash(output / name)
        for name in ["resnet50.pt", "densenet121.pt", "text_model.joblib"]
    }
    manifest = {
        "schema_version": 2,
        "kind": "team29_late",
        "model_version": args.model_version or "0.2.0-team29-" + selection["cohort_manifest"][:8],
        "preprocessing_version": PREPROCESS,
        "image_size": 384,
        "classes": ["proxy_negative", "proxy_abnormal"],
        "text_fields": ["indication", "comparison"],
        "required_text_fields": ["indication"],
        "threshold": selection["thresholds"]["multimodal_fusion"],
        "image_weight": selection["image_weight"],
        "calibrated": False,
        "cohort_manifest": selection["cohort_manifest"],
        "selection": selection["objective"],
        "evaluation_unit": "study: mean image scores; UI analyzes one image at a time",
        "gradcam_target": "ResNet50 branch of the final image-ensemble/text score",
        "cohort_label_version": "team29-strict-core-tags-v1; broader thoracic weak labels",
        "library_versions": {
            name: importlib.metadata.version(name)
            for name in ["torch", "torchvision", "scikit-learn", "numpy", "pandas"]
        },
        "files": files,
    }
    save_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
