"""Checks for grouping, weak labels, negation, and transform geometry."""

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from pulmonary.evaluation import choose_threshold, study_predictions
from pulmonary.images import prepare_pil
from pulmonary.labels import model_text, weak_label
from pulmonary.priority import get_priority


def test_weak_labels_are_conservative_and_conflicts_are_uncertain():
    assert weak_label("normal", "normal") == "normal"
    assert weak_label("normal", "pneumonia") == "uncertain"
    assert weak_label("atelectasis", "lung") == "abnormal"
    assert weak_label("scarring", "") == "uncertain"
    assert weak_label("", "") == "uncertain"
    assert weak_label("massive", "") == "uncertain"


def test_prediction_text_excludes_targets_and_preserves_negation():
    row = {
        "indication": "No dyspnea. History of pneumonia.",
        "findings": "Target leakage",
        "problems": "normal",
    }
    text = model_text(row)
    assert "no dyspnea" in text
    assert "pneumonia" in text  # A pre-examination history is not automatically target leakage.
    assert "leakage" not in text
    with pytest.raises(ValueError):
        model_text(row, fields=["findings"])


def test_transform_preserves_full_rectangular_image():
    image = Image.new("L", (400, 100), 255)
    prepared = np.asarray(prepare_pil(image, 224))
    assert prepared.shape == (224, 224, 3)
    assert prepared[112, 0].min() == 255
    assert prepared[112, -1].min() == 255
    assert prepared[0].max() == 0
    assert prepared[-1].max() == 0


def test_studies_not_image_rows_are_primary_units():
    frame = pd.DataFrame({"uid": ["a", "a", "b"], "target": [0, 0, 1]})
    result = study_predictions(frame, [0.1, 0.3, 0.8])
    assert len(result) == 2
    assert result.loc[result.uid.eq("a"), "probability"].item() == pytest.approx(0.2)
    threshold, table = choose_threshold(frame, [0.1, 0.3, 0.8], {"fn_cost": 10, "fp_cost": 1})
    assert 0.2 < threshold <= 0.8
    assert table.cost.min() == 0


def test_priority_rejects_nonfinite_and_does_not_claim_urgency():
    for value in [float("nan"), float("inf"), -0.1, 1.1]:
        with pytest.raises(ValueError):
            get_priority(value, 0.4)
    assert get_priority(0.1, 0.4)[0] == "lower_review"
    assert get_priority(0.5, 0.4)[0] == "review"
    assert get_priority(0.9, 0.4)[0] == "higher_review"
