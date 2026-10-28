"""Run the FastAPI service inside the Streamlit process for single-container hosting.

Requests still pass through the same routes, validation and response schemas as the
HTTP service. A remote bundle is accepted only if the whole archive matches a pinned
SHA-256; the engine then verifies each artifact against the bundle manifest.
"""

import hashlib
import logging
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from fastapi.testclient import TestClient

from app.main import create_app

LOGGER = logging.getLogger(__name__)
ARCHIVE_PREFIX = "artifacts/team_bundle/"
BUNDLE_FILES = {"manifest.json", "resnet50.pt", "densenet121.pt", "text_model.joblib"}


def fetch_bundle(url, sha256, cache_root):
    """Download and unpack a pinned bundle archive once; return the bundle folder."""
    if len(sha256) != 64:
        raise ValueError("A 64-character SHA-256 is required for a downloaded bundle")
    folder = Path(cache_root) / sha256
    if (folder / "manifest.json").exists():
        return folder
    Path(cache_root).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache_root) as temporary:
        archive = Path(temporary) / "bundle.zip"
        digest = hashlib.sha256()
        with urllib.request.urlopen(url, timeout=300) as response, archive.open("wb") as out:
            for block in iter(lambda: response.read(1024 * 1024), b""):
                digest.update(block)
                out.write(block)
        if digest.hexdigest() != sha256:
            raise ValueError("Downloaded model bundle does not match the pinned SHA-256")
        staging = Path(temporary) / "bundle"
        staging.mkdir()
        with ZipFile(archive) as source:
            for info in source.infolist():
                name = PurePosixPath(info.filename)
                if info.filename.startswith(ARCHIVE_PREFIX) and name.name in BUNDLE_FILES:
                    with source.open(info) as stream, (staging / name.name).open("wb") as out:
                        shutil.copyfileobj(stream, out)
        missing = BUNDLE_FILES - {p.name for p in staging.iterdir()}
        if missing:
            raise ValueError(f"Model bundle archive is missing {sorted(missing)}")
        staging.replace(folder)
    return folder


def resolve_bundle():
    """Use a local bundle when present, otherwise the pinned download if configured."""
    folder = Path(os.environ.get("MODEL_BUNDLE", "artifacts/team_bundle"))
    url = os.environ.get("MODEL_BUNDLE_URL")
    if (folder / "manifest.json").exists() or not url:
        return folder
    cache = os.environ.get("MODEL_BUNDLE_CACHE", Path(tempfile.gettempdir()) / "pulmonary")
    return fetch_bundle(url, os.environ.get("MODEL_BUNDLE_SHA256", ""), cache)


class InProcessService:
    """Same interface the UI uses for HTTP: get/post returning status and JSON."""

    def __init__(self, application=None):
        if application is None:
            try:
                os.environ["MODEL_BUNDLE"] = str(resolve_bundle())
            except (OSError, ValueError):
                # The service then reports not_ready, exactly like a missing local bundle.
                LOGGER.exception("Model bundle download failed")
            application = create_app()
        self.client = TestClient(application, raise_server_exceptions=False)
        self.client.__enter__()  # runs the lifespan once, loading the engine

    def get(self, path, **kwargs):
        return self.client.get(path, **kwargs)

    def post(self, path, **kwargs):
        return self.client.post(path, **kwargs)
