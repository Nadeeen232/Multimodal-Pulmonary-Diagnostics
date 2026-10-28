"""Exercise real Torch fusion, export, checksums, API, and both explanations.

Random weights and tiny artificial text are a software fixture only. They are
never written under the project's real artifact tree or reported as model quality.
"""

import io

import joblib
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from app.main import create_app
from pulmonary.common import file_hash, save_json
from pulmonary.explainability import explain
from pulmonary.images import PREPROCESS_VERSION
from pulmonary.inference import InferenceEngine
from pulmonary.networks import FusionHead, ImageEncoder


def test_real_early_fusion_explanation_and_checksum_validation(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    encoder = ImageEncoder("resnet50", pretrained=False)
    torch.save(
        {
            "state_dict": encoder.state_dict(),
            "backbone": "resnet50",
            "dropout": 0.5,
            "image_size": 64,
        },
        tmp_path / "backbone.pt",
    )
    vectorizer = TfidfVectorizer().fit(["indication no cough", "indication dyspnea"])
    features = encoder.feature_dim + len(vectorizer.get_feature_names_out())
    scaler = StandardScaler().fit(np.vstack([np.zeros(features), np.ones(features)]))
    fusion = FusionHead(features, hidden=8)
    torch.save(
        {"state_dict": fusion.state_dict(), "input_dim": features, "hidden": 8, "dropout": 0.5},
        tmp_path / "fusion.pt",
    )
    joblib.dump(vectorizer, tmp_path / "vectorizer.joblib")
    joblib.dump(scaler, tmp_path / "scaler.joblib")
    names = ["backbone.pt", "fusion.pt", "vectorizer.joblib", "scaler.joblib"]
    manifest = {
        "schema_version": 1,
        "preprocessing_version": PREPROCESS_VERSION,
        "image_size": 64,
        "kind": "early",
        "text_fields": ["indication"],
        "strip_target_terms": False,
        "threshold": 0.5,
        "model_version": "software-test-only",
        "files": {name: file_hash(tmp_path / name) for name in names},
    }
    save_json(tmp_path / "manifest.json", manifest)
    engine = InferenceEngine(tmp_path)
    image = Image.fromarray(
        np.random.default_rng(42).integers(0, 256, size=(64, 128), dtype=np.uint8)
    )
    prediction = engine.predict(image, "no cough")
    result = explain(engine, image, "no cough")
    assert result["prediction"]["probability"] == pytest.approx(prediction["probability"])
    assert len(result["gradcam_base64"]) > 100
    assert {v["token"] for v in result["text_highlights"]} == {"no", "cough"}
    assert result["text_tokens_truncated"] is False
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    with TestClient(create_app(lambda: engine)) as client:
        response = client.post(
            "/explain",
            files={"image": ("fixture.png", buffer.getvalue(), "image/png")},
            data={"indication": "no cough"},
        )
        assert response.status_code == 200, response.text
        assert response.json()["prediction"]["model_version"] == "software-test-only"
    with (tmp_path / "fusion.pt").open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        InferenceEngine(tmp_path)
