import numpy as np
import pandas as pd
import pytest
from PIL import Image

from pulmonary.learned_reports import LearnedReportGenerator
from pulmonary.team_classification import letterbox, study_scores, threshold
from pulmonary.team_rules import make_report_text


def test_team_text_does_not_read_current_report_or_label_tags():
    row = {
        "indication": "XXXX shortness of breath",
        "comparison": "None.",
        "findings": "large mass",
        "impression": "abnormal",
        "Problems": "Mass",
    }
    before = make_report_text(row)
    row.update(findings="clear", impression="normal", Problems="normal")
    assert make_report_text(row) == before
    assert "shortness of breath" in before
    assert "xxxx" not in before


def test_letterbox_preserves_aspect_and_uses_black_padding():
    image = Image.new("L", (400, 200), color=200)
    result = np.asarray(letterbox(image))
    assert result.shape == (384, 384)
    assert not result[:96].any()
    assert np.all(result[96:288] == 200)
    assert not result[288:].any()


def test_study_averaging_rejects_conflicting_labels():
    frame = pd.DataFrame(
        {"uid": ["a", "a", "b"], "target": [1, 1, 0], "probability": [0.2, 0.8, 0.1]}
    )
    assert study_scores(frame).loc["a", "probability"] == pytest.approx(0.5)
    frame.loc[1, "target"] = 0
    with pytest.raises(ValueError, match="Conflicting"):
        study_scores(frame)


def test_threshold_uses_validation_balanced_accuracy():
    y = np.array([0, 0, 0, 1])
    p = np.array([0.1, 0.2, 0.3, 0.4])
    chosen = threshold(y, p)
    assert 0.3 < chosen <= 0.4


def test_generator_rejects_missing_import_checksums(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="checksums"):
        LearnedReportGenerator(tmp_path)


def test_team_bundle_serves_matching_fusion_with_explanations(tmp_path):
    """Random weights verify software parity only; no model-quality claim."""
    import io

    import joblib
    import torch
    from fastapi.testclient import TestClient
    from sklearn.linear_model import LogisticRegression

    from app.main import create_app
    from pulmonary.common import file_hash, save_json
    from pulmonary.explainability import explain
    from pulmonary.inference import load_engine
    from pulmonary.team_features import PREPROCESS, TeamTextModel, build_model

    torch.set_num_threads(2)
    torch.manual_seed(42)
    frame = pd.DataFrame(
        {"indication": ["severe cough"] * 6 + ["preop screening"] * 6, "comparison": ["none"] * 12}
    )
    text = TeamTextModel()
    text.classifier = LogisticRegression().fit(text.matrix(frame, fit=True), [1] * 6 + [0] * 6)
    joblib.dump(text, tmp_path / "text_model.joblib")
    nets = []
    for name in ["resnet50", "densenet121"]:
        net = build_model(name).eval()
        torch.save(
            {
                "state_dict": net.state_dict(),
                "cohort_manifest": "software-fixture",
                "preprocess": PREPROCESS,
            },
            tmp_path / f"{name}.pt",
        )
        nets.append(net)
    manifest = {
        "kind": "team29_late",
        "preprocessing_version": PREPROCESS,
        "image_size": 384,
        "threshold": 0.5,
        "image_weight": 0.7,
        "cohort_manifest": "software-fixture",
        "model_version": "software-fixture-only",
        "text_fields": ["indication", "comparison"],
        "required_text_fields": ["indication"],
        "files": {
            n: file_hash(tmp_path / n)
            for n in ["resnet50.pt", "densenet121.pt", "text_model.joblib"]
        },
    }
    save_json(tmp_path / "manifest.json", manifest)
    engine = load_engine(tmp_path)
    image = Image.fromarray(
        np.random.default_rng(42).integers(0, 255, size=(64, 96), dtype=np.uint8)
    )
    tensor = engine.image_tensor(image).unsqueeze(0)
    with torch.inference_mode():
        image_score = sum(float(net(tensor).softmax(1)[0, 1]) for net in nets) / 2
        text_score = float(
            text.predict_proba(pd.DataFrame([{"indication": "severe cough", "comparison": ""}]))[
                0, 1
            ]
        )
    prediction = engine.predict(image, "severe cough")
    assert prediction["probability"] == pytest.approx(
        0.7 * image_score + 0.3 * text_score, abs=1e-6
    )
    result = explain(engine, image, "severe cough", max_tokens=2)
    assert result["prediction"]["probability"] == pytest.approx(prediction["probability"])
    assert len(result["gradcam_base64"]) > 100
    assert result["generated_report"]
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    with TestClient(create_app(lambda: engine)) as client:
        response = client.post(
            "/predict",
            files={"image": ("fixture.png", buffer.getvalue(), "image/png")},
            data={"indication": "severe cough"},
        )
        assert response.status_code == 200, response.text
        assert response.json()["probability"] == pytest.approx(prediction["probability"])
    with (tmp_path / "text_model.joblib").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        load_engine(tmp_path)
