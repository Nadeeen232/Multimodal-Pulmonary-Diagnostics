"""Check UI startup and required-input behavior without a trained backend."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest


def test_ui_starts_and_requires_both_inputs(monkeypatch):
    monkeypatch.setenv("API_URL", "http://127.0.0.1:1")
    page = AppTest.from_file(
        Path(__file__).resolve().parents[1] / "ui/app.py", default_timeout=20
    ).run()
    assert not page.exception
    assert any("Clinician verification is required" in message.value for message in page.warning)
    assert all("educational" not in message.value.lower() for message in page.warning)
    assert page.button[0].disabled
