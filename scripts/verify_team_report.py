"""Verify the imported learned report model on saved validation-study features."""

import argparse
import os
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import json

import numpy as np
import pandas as pd
import torch

from pulmonary.common import file_hash, save_json
from pulmonary.learned_reports import LearnedReportGenerator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="artifacts/team_import")
    parser.add_argument("--run", default="20261026_180944_731991")
    parser.add_argument("--examples", type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = Path(args.root)
    source = root / "v1_source"
    folder = root / "report_generation_v2" / args.run / "candidate_generator"
    generator = LearnedReportGenerator(folder)
    for name, expected in generator.config["source_manifest_hashes"].items():
        if file_hash(source / name) != expected:
            raise ValueError(f"Source manifest mismatch: {name}")
    parts = {
        s: json.loads((source / f"{s}_studies.json").read_text()) for s in ["train", "val", "test"]
    }
    ids = {s: {row["uid"] for row in rows} for s, rows in parts.items()}
    if ids["train"] & ids["val"] or ids["train"] & ids["test"] or ids["val"] & ids["test"]:
        raise ValueError("Report study split overlap")
    audit = pd.read_csv(source / "image_audit.csv", dtype=str)
    if audit.groupby("sha256").split.nunique().gt(1).any():
        raise ValueError("Report exact-image split overlap")
    with np.load(source / "image_features.npz", allow_pickle=False) as archive:
        features = dict(zip(archive["names"].astype(str), archive["vectors"], strict=True))
    names = np.load(root / "xrv_selected/names.npy", allow_pickle=False).astype(str)
    score_values = np.load(root / "xrv_selected/scores.npy", allow_pickle=False)
    pathologies = (
        np.load(root / "xrv_selected/pathologies.npy", allow_pickle=False).astype(str).tolist()
    )
    if pathologies != generator.config["tags"]["pathologies"]:
        raise ValueError("XRV pathology order mismatch")
    scores = dict(zip(names, score_values, strict=True))
    results = []
    for row in parts["val"][: args.examples]:
        prediction = generator.generate_features(
            np.stack([features[n] for n in row["image_names"]]),
            row["view_ids"],
            row["indication"],
            np.stack([scores[n] for n in row["image_names"]]),
        )
        results.append({"uid": row["uid"], "reference": row["target_report"], **prediction})
        print(json.dumps(results[-1], indent=2), flush=True)
    expected_first = "findings: the heart size and mediastinal contours are within normal limits. there is no focal consolidation, pleural effusion, or pneumothorax. the lungs are clear. impression: no acute cardiopulmonary abnormality."
    verified = bool(results) and results[0]["report"] == expected_first
    result = {
        "run": args.run,
        "split": "validation",
        "examples": results,
        "matches_notebook_first_example": verified,
        "source_manifest_hashes_verified": True,
        "uid_overlap": False,
        "exact_image_overlap": False,
        "counts": {s: len(rows) for s, rows in parts.items()},
        "scope": "Export reload and example verification; not a new full-cohort clinical evaluation",
    }
    save_json("reports/generated/team_review/report_reload_verification.json", result)
    if not verified:
        raise RuntimeError(
            "Imported generator differs from the notebook reload example; investigate before use"
        )


if __name__ == "__main__":
    main()
