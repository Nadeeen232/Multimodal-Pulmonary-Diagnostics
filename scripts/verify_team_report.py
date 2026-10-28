"""Verify the imported final learned report model on saved validation-study features.

The export comes from notebooks/Team29_Report_Generation_Final.ipynb (section 09).
"""

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
    parser.add_argument("--run", default="20261028_113222_436989")
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
    weights = generator.config["tags"]["xrv_weights"]
    with np.load(root / f"xrv_pooled_{weights.split('-')[-1]}.npz", allow_pickle=False) as archive:
        if str(archive["weights"]) != weights:
            raise ValueError("Pooled XRV features were built with different weights")
        pooled = dict(zip(archive["names"].astype(str), archive["pooled"], strict=True))
    results = []
    for row in parts["val"][: args.examples]:
        prediction = generator.generate_features(
            np.stack([features[n] for n in row["image_names"]]),
            row["view_ids"],
            row["indication"],
            np.stack([pooled[n] for n in row["image_names"]]),
        )
        results.append({"uid": row["uid"], "reference": row["target_report"], **prediction})
        print(json.dumps(results[-1], indent=2), flush=True)
    # Printed by the notebook's reload check (section 09) for the first validation study.
    expected_prompt = "generate chest x-ray report. indication: preop chest x- .. image findings: degenerative spine"
    expected_first = "findings: the cardiomediastinal silhouette is within normal limits for size and contour. the lungs are clear. no focal consolidation, pneumothorax, or pleural effusion. impression: no acute cardiopulmonary abnormality."
    verified = (
        bool(results)
        and results[0]["prompt"] == expected_prompt
        and results[0]["report"] == expected_first
    )
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
