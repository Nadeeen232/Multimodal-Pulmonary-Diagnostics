"""Small image/metadata fixtures exercise the real preparation path."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from pulmonary.common import load_splits
from pulmonary.data import link_duplicate_groups, prepare


def test_duplicate_linkage_is_transitive_across_studies():
    frame = pd.DataFrame(
        {
            "group_id": ["a", "b", "b", "c", "d"],
            "image_sha256": ["one", "one", "two", "two", "three"],
        }
    )
    linked, count = link_duplicate_groups(frame)
    assert linked.group_id.tolist() == ["a", "a", "a", "a", "d"]
    assert count == 4


def test_real_preparation_groups_images_and_detects_split_tampering(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    reports, projections = [], []
    for index in range(40):
        label = "normal" if index % 2 == 0 else "pneumonia"
        reports.append(
            {
                "uid": str(index),
                "problems": label,
                "mesh": label,
                "indication": "no cough" if index % 2 == 0 else "cough",
            }
        )
        for view in [0, 1]:
            name = f"{index}_{view}.png"
            projections.append({"uid": str(index), "filename": name})
            image = Image.fromarray(
                np.random.default_rng(index * 2 + view).integers(
                    0, 256, size=(32, 64), dtype=np.uint8
                )
            )
            image.save(raw / name)
    pd.DataFrame(reports).to_csv(raw / "indiana_reports.csv", index=False)
    pd.DataFrame(projections).to_csv(raw / "indiana_projections.csv", index=False)
    config = {
        "data_root": str(raw),
        "output_root": str(tmp_path / "artifacts"),
        "reports_root": str(tmp_path / "reports"),
        "seed": 42,
        "text_fields": ["indication"],
        "strip_target_terms": False,
        "image_size": 64,
    }
    audit = prepare(config)
    assert audit["image_validation"] == "complete"
    splits = load_splits(config, include_test=True)
    assert sum(len(frame) for frame in splits.values()) == 80
    assert sum(frame.uid.nunique() for frame in splits.values()) == 40
    for frame in splits.values():
        assert all(
            (Path(config["output_root"]) / path).exists() for path in frame.processed_image_path
        )
    path = Path(config["output_root"]) / "splits/train.csv"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="changed"):
        load_splits(config)
