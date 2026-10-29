"""Check UI startup and required-input behavior without a trained backend."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "ui/app.py"


def analyze_button(page):
    return next(button for button in page.button if button.label == "Analyze Study")


def test_ui_starts_and_requires_both_inputs(monkeypatch):
    monkeypatch.setenv("API_URL", "http://127.0.0.1:1")
    page = AppTest.from_file(APP, default_timeout=20).run()
    assert not page.exception
    assert any("Clinician verification is required" in c.value for c in page.caption)
    assert all("educational" not in m.value.lower() for m in [*page.caption, *page.markdown])
    assert any("prediction service is unavailable" in m.value for m in page.markdown)
    assert analyze_button(page).disabled


def test_ui_does_not_shadow_the_app_package(monkeypatch):
    """ui/app.py must not put ui/ on sys.path, or `import app.inprocess` breaks on Cloud."""
    import sys

    monkeypatch.setenv("API_URL", "http://127.0.0.1:1")
    AppTest.from_file(APP, default_timeout=20).run()
    assert str(APP.parent) not in sys.path


def test_sample_case_fills_the_study(monkeypatch):
    monkeypatch.setenv("API_URL", "http://127.0.0.1:1")
    page = AppTest.from_file(APP, default_timeout=20).run()
    sample = next(button for button in page.button if button.label == "Abnormal")
    page = sample.click().run()
    assert not page.exception
    assert page.text_area(key="indication").value == "Short of breath."
    assert any("Test study 2348" in m.value for m in page.markdown)
    # The service is unreachable in this test, so Analyze stays disabled.
    assert analyze_button(page).disabled


def test_result_renders_verdict_heatmap_tokens_and_report(monkeypatch):
    import base64
    import hashlib
    import io

    from PIL import Image

    monkeypatch.setenv("API_URL", "http://127.0.0.1:1")
    png = io.BytesIO()
    Image.new("RGB", (8, 8)).save(png, format="PNG")
    report = (
        "Predicted Class: Abnormal\n\nImpression:\nScore above threshold.\n\n"
        "Recommendation:\nClinician verification is required."
    )
    result = {
        "predicted_class": "Abnormal",
        "prediction": {
            "probability": 0.905,
            "threshold": 0.35,
            "priority_tier": "higher_review",
            "priority_reason": "Model score meets the illustrative higher-review cutoff (0.675).",
            "model_version": "test-version",
        },
        "gradcam_base64": base64.b64encode(png.getvalue()).decode(),
        "text_highlights": [
            {"field": "indication", "token": "breath", "start": 0, "end": 6, "contribution": 0.0087}
        ],
        "method": "occlusion",
        "text_tokens_truncated": False,
        "generated_report": report,
    }
    page = AppTest.from_file(APP, default_timeout=20)
    page.session_state["analysis"] = result
    # Matches the empty study, so the result is not cleared as stale.
    page.session_state["analysis_input"] = hashlib.sha256(b"\x00").hexdigest()
    page.run()
    assert not page.exception
    markdown = " ".join(m.value for m in page.markdown)
    assert '90<span class="pt">.</span>5' in markdown
    assert "At or above the 35% threshold" in markdown
    assert "breath" in markdown and "+0.87" in markdown
    assert any(c.value == report for c in page.code)
    assert {"Download .txt", "Download .json"} <= {b.label for b in page.get("download_button")}


def test_header_uses_the_app_icon(monkeypatch):
    monkeypatch.setenv("API_URL", "http://127.0.0.1:1")
    page = AppTest.from_file(APP, default_timeout=20).run()
    assert any('class="pr-logo" src="data:image/png;base64,' in m.value for m in page.markdown)
