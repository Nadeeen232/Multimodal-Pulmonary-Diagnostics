"""Create three versioned, selected demo cases from the completed test outputs."""

import os
import re
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import shutil

import pandas as pd
import torch
from PIL import Image

from pulmonary.common import file_hash, read_config
from pulmonary.inference import load_engine
from pulmonary.team_rules import make_report_text


def main():
    torch.set_num_threads(2)
    source = Path("artifacts/team_classification")
    root = Path(read_config("configs/local.yaml")["data_root"])
    engine = load_engine("artifacts/team_bundle")
    resnet = pd.read_csv(source / "resnet50_test_predictions.csv", dtype={"uid": str}).rename(
        columns={"probability": "p_resnet"}
    )
    dense = pd.read_csv(source / "densenet121_test_predictions.csv", dtype={"uid": str})
    frame = resnet.merge(
        dense[["uid", "filename", "probability"]].rename(columns={"probability": "p_dense"}),
        on=["uid", "filename"],
        validate="one_to_one",
    )
    if len(frame) != len(resnet) or len(frame) != len(dense):
        raise ValueError("Incomplete image prediction alignment")
    studies = pd.read_csv(source / "studies.csv", dtype={"uid": str}, keep_default_na=False)
    usable = studies.indication.map(
        lambda value: (
            len(re.findall(r"[a-z]{2,}", make_report_text({"indication": value, "comparison": ""})))
            >= 2
        )
    )
    probabilities = pd.read_csv(source / "study_test_predictions.csv", dtype={"uid": str})
    frame = frame.merge(probabilities[["uid", "p_text_only"]], on="uid", validate="many_to_one")
    frame = frame.loc[frame.uid.isin(studies.loc[usable, "uid"])].copy()
    alpha = engine.manifest["image_weight"]
    cutoff = engine.manifest["threshold"]
    frame["fused"] = alpha * (frame.p_resnet + frame.p_dense) / 2 + (1 - alpha) * frame.p_text_only
    frame["predicted"] = frame.fused.ge(cutoff).astype(int)
    choices = [
        frame.loc[
            frame.target.eq(0) & frame.predicted.eq(0) & frame.fused.lt(min(0.25, cutoff - 0.1))
        ],
        frame.loc[
            frame.target.eq(1) & frame.predicted.eq(1) & frame.fused.gt(max(0.7, cutoff + 0.1))
        ],
        frame.loc[frame.target.ne(frame.predicted)],
    ]
    if any(part.empty for part in choices):
        raise ValueError("Cannot select all three requested demo categories from these predictions")
    output = Path("demo_images/team29")
    output.mkdir(parents=True, exist_ok=True)
    current = Path("demo_images/demo_cases.csv")
    previous = Path("demo_images/demo_cases_v1.csv")
    if current.exists() and not previous.exists():
        shutil.copyfile(current, previous)
    rows = []
    for number, part in enumerate(choices, 1):
        # Samples are selected after evaluation for demonstration, never for metrics.
        row = part.sample(1, random_state=41 + number).iloc[0]
        study = studies.loc[studies.uid.eq(row.uid)].iloc[0]
        matches = list(root.rglob(row.filename))
        if len(matches) != 1 or file_hash(matches[0]) != row.image_sha256:
            raise ValueError("Demo source image does not match the prepared cohort")
        with Image.open(matches[0]) as image:
            prediction = engine.predict(image, study.indication, study.comparison)
        predicted = int(prediction["predicted_label"] == "proxy_abnormal")
        outcome = {
            (0, 0): "true_negative",
            (1, 1): "true_positive",
            (0, 1): "false_positive",
            (1, 0): "false_negative",
        }[(int(row.target), predicted)]
        filename = f"{number:02d}_{outcome}_uid_{row.uid}.png"
        shutil.copyfile(matches[0], output / filename)
        note_name = filename.replace(".png", "_context.txt")
        (output / note_name).write_text(
            f"Indication: {study.indication}\nComparison: {study.comparison}\n", encoding="utf-8"
        )
        rows.append(
            {
                "category": ["normal", "abnormal", "false_prediction"][number - 1],
                "uid": row.uid,
                "filename": "team29/" + filename,
                "indication_file": "team29/" + note_name,
                "source_path": matches[0].relative_to(root).as_posix(),
                "source_sha256": row.image_sha256,
                "weak_label": "abnormal" if row.target else "normal",
                **prediction,
                "outcome": outcome,
                "indication": study.indication,
                "comparison": study.comparison,
                "split": "test",
                "evaluation_unit": "single image; selected after test evaluation",
            }
        )
    pd.DataFrame(rows).to_csv(current, index=False)
    pd.DataFrame(rows).to_csv(output / "demo_cases.csv", index=False)
    print(
        pd.DataFrame(rows)[["uid", "filename", "weak_label", "outcome", "probability"]].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
