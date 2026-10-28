"""Report semantics must preserve uncertainty and attribution direction."""

import numpy as np
import pytest

from pulmonary.explainability import describe_region
from report_generator import generate_report


def test_report_preserves_direction_and_threshold_priority():
    report = generate_report(
        0.135,
        "review",
        "upper image region",
        [
            {"token": "cough", "contribution": 0.03},
            {"token": "no", "contribution": -0.02},
        ],
    )
    assert report.startswith("Predicted Class: Abnormal")
    assert "13.5%" in report
    assert "'cough' were associated with a higher" in report
    assert "'no' were associated with a lower" in report
    assert all(
        section in report
        for section in [
            "Impression:",
            "Visual Findings:",
            "Clinical Context:",
            "Recommendation:",
        ]
    )
    assert "educational" not in report.lower()
    assert "Clinician verification is required" in report
    negative = generate_report(0.01, "lower_review", None, [])
    assert negative.startswith("Predicted Class: Normal")
    assert "does not exclude disease" in negative
    assert "No informative text attribution" in negative
    assert "No interpretable positive Grad-CAM" in negative
    with pytest.raises(ValueError):
        generate_report(float("nan"), "review", None, [])


def test_region_description_handles_localized_diffuse_and_empty_maps():
    cam = np.zeros((9, 9))
    cam[:3, :3] = 1
    assert describe_region(cam).startswith("upper image-left")
    assert "diffuse" in describe_region(np.ones((9, 9)))
    assert "unavailable" in describe_region(np.zeros((9, 9)))
