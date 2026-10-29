"""Single-container hosting must serve the same API and accept only the pinned bundle."""

import hashlib
from pathlib import Path
from zipfile import ZipFile

import pytest

from app.inprocess import BUNDLE_FILES, InProcessService, fetch_bundle
from app.main import create_app
from tests.test_api import TestEngine, png

ROOT = Path(__file__).resolve().parents[1]


def bundle_zip(path, files=BUNDLE_FILES):
    with ZipFile(path, "w") as archive:
        for name in files:
            archive.writestr("artifacts/team_bundle/" + name, name.encode())
        archive.writestr("MODEL_BUNDLE_README.txt", "readme")
        archive.writestr("artifacts/other/resnet50.pt", "not part of the bundle")
    return path.as_uri(), hashlib.sha256(path.read_bytes()).hexdigest()


def test_fetch_unpacks_only_bundle_files_and_reuses_the_cache(tmp_path):
    url, digest = bundle_zip(tmp_path / "bundle.zip")
    folder = fetch_bundle(url, digest, tmp_path / "cache")
    assert {p.name for p in folder.iterdir()} == BUNDLE_FILES
    assert (folder / "resnet50.pt").read_bytes() == b"resnet50.pt"
    (tmp_path / "bundle.zip").unlink()
    assert fetch_bundle(url, digest, tmp_path / "cache") == folder


def test_fetch_rejects_unpinned_or_mismatched_archives(tmp_path):
    url, digest = bundle_zip(tmp_path / "bundle.zip")
    with pytest.raises(ValueError, match="64-character"):
        fetch_bundle(url, "", tmp_path / "cache")
    with pytest.raises(ValueError, match="does not match"):
        fetch_bundle(url, "0" * 64, tmp_path / "cache")
    assert not (tmp_path / "cache" / ("0" * 64)).exists()


def test_fetch_rejects_incomplete_archives(tmp_path):
    url, digest = bundle_zip(tmp_path / "bundle.zip", BUNDLE_FILES - {"densenet121.pt"})
    with pytest.raises(ValueError, match="densenet121.pt"):
        fetch_bundle(url, digest, tmp_path / "cache")


def test_in_process_service_matches_http_api_responses(monkeypatch):
    from report_generator import generate_report

    def fixture_explain(engine, image, indication, comparison):
        return {
            "prediction": engine.predict(image, indication, comparison),
            "predicted_class": "Abnormal",
            "gradcam_region": "unavailable (test fixture)",
            "generated_report": generate_report(0.7, "higher_review", None, []),
        }

    monkeypatch.setattr("app.main.explain", fixture_explain)
    service = InProcessService(create_app(engine_factory=TestEngine))
    assert service.get("/health").json() == {"status": "ready", "model_version": "test-only"}
    response = service.post(
        "/predict",
        files={"image": ("x.png", png(), "image/png")},
        data={"indication": "cough"},
    )
    assert response.status_code == 200
    assert response.json()["predicted_class"] == "Abnormal"
    rejected = service.post(
        "/predict", files={"image": ("x.png", png(), "image/png")}, data={"indication": " "}
    )
    assert rejected.status_code == 422


def test_failed_download_reports_not_ready(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_BUNDLE", str(tmp_path / "absent"))
    monkeypatch.setenv("MODEL_BUNDLE_URL", (tmp_path / "missing.zip").as_uri())
    monkeypatch.setenv("MODEL_BUNDLE_SHA256", "0" * 64)
    monkeypatch.setenv("MODEL_BUNDLE_CACHE", str(tmp_path / "cache"))
    service = InProcessService()
    assert service.get("/health").json() == {"status": "not_ready", "model_version": None}


def test_cloud_entrypoint_runs_ui_in_process(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("MODEL_BUNDLE", str(tmp_path / "absent"))
    monkeypatch.setenv("MODEL_BUNDLE_URL", (tmp_path / "missing.zip").as_uri())
    monkeypatch.setenv("MODEL_BUNDLE_CACHE", str(tmp_path / "cache"))
    # Set (not only default) everything the entrypoint defaults, so it is restored after.
    for name, value in [
        ("PULMONARY_IN_PROCESS", "1"),
        ("MODEL_BUNDLE_SHA256", "0" * 64),
        ("OMP_NUM_THREADS", "2"),
        ("MKL_NUM_THREADS", "2"),
    ]:
        monkeypatch.setenv(name, value)
    st.cache_resource.clear()
    page = AppTest.from_file(
        str(ROOT / "deploy/streamlit/streamlit_app.py"), default_timeout=60
    ).run()
    st.cache_resource.clear()
    assert not page.exception
    assert any("model is not ready" in message.value for message in page.markdown)
    assert next(button for button in page.button if button.label == "Analyze").disabled
