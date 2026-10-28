"""API validation uses an explicit test double, never a shipped fake model."""

import io

from fastapi.testclient import TestClient
from PIL import Image

from app.main import create_app


class TestEngine:
    __test__ = False
    manifest = {"model_version": "test-only", "text_fields": ["indication"]}

    def text(self, indication, comparison=""):
        return indication.strip()

    def predict(self, image, indication, comparison=""):
        assert image.size == (32, 32)
        return {
            "probability": 0.7,
            "predicted_label": "proxy_abnormal",
            "threshold": 0.4,
            "priority_tier": "higher_review",
            "priority_reason": "Test fixture",
            "model_version": "test-only",
            "calibrated": False,
        }


def png():
    buffer = io.BytesIO()
    Image.new("L", (32, 32), 128).save(buffer, format="PNG")
    return buffer.getvalue()


def test_missing_model_has_liveness_but_no_readiness_or_prediction(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_BUNDLE", str(tmp_path / "absent"))
    with TestClient(create_app()) as client:
        assert client.get("/health").json() == {"status": "not_ready", "model_version": None}
        assert client.get("/ready").status_code == 503
        assert (
            client.post(
                "/predict",
                files={"image": ("x.png", png(), "image/png")},
                data={"indication": "cough"},
            ).status_code
            == 503
        )


def test_valid_multipart_prediction_and_validation_errors(monkeypatch):
    from report_generator import generate_report

    def fixture_explain(engine, image, indication, comparison):
        prediction = engine.predict(image, indication, comparison)
        return {
            "prediction": prediction,
            "predicted_class": "Abnormal",
            "gradcam_region": "unavailable (test fixture)",
            "generated_report": generate_report(0.7, "higher_review", None, []),
        }

    monkeypatch.setattr("app.main.explain", fixture_explain)
    with TestClient(create_app(TestEngine)) as client:
        files = {"image": ("x.png", png(), "image/png")}
        response = client.post("/predict", files=files, data={"indication": "cough"})
        assert response.status_code == 200
        assert response.json()["model_version"] == "test-only"
        assert response.json()["predicted_class"] == "Abnormal"
        assert "Impression:" in response.json()["generated_report"]
        assert client.post("/predict", files=files, data={"indication": "  "}).status_code == 422
        assert (
            client.post(
                "/predict",
                files={"image": ("bad.png", b"not a picture", "image/png")},
                data={"indication": "cough"},
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/predict",
                files={"image": ("x.dcm", png(), "application/dicom")},
                data={"indication": "cough"},
            ).status_code
            == 415
        )
        assert client.post("/predict", data={"indication": "cough"}).status_code == 422
        assert client.get("/metrics").json()["requests"] >= 5


def test_request_size_limit():
    with TestClient(create_app(TestEngine)) as client:
        response = client.post(
            "/predict", content=b"", headers={"content-length": str(11 * 1024 * 1024)}
        )
        assert response.status_code == 413
