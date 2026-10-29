"""Check UI startup and required-input behavior without a trained backend."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "ui/app.py"


def analyze_button(page):
    return next(button for button in page.button if button.label == "Analyze")


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
    sample = next(button for button in page.button if button.label == "Abnormal case")
    page = sample.click().run()
    assert not page.exception
    assert page.text_area(key="indication").value == "Short of breath."
    assert any("Test study 2348" in c.value for c in page.caption)
    # The service is unreachable in this test, so Analyze stays disabled.
    assert analyze_button(page).disabled
