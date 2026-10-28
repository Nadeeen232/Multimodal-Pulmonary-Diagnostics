"""Streamlit Community Cloud entrypoint: the UI with the model service in-process.

Community Cloud runs one process, so ui/app.py starts the FastAPI app internally
instead of calling it over HTTP. The model bundle is downloaded from the GitHub
release below and accepted only if the archive matches the pinned SHA-256.
Any of these defaults can be overridden with app secrets of the same name.
"""

import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [p for p in (str(ROOT / "src"), str(ROOT)) if p not in sys.path]

os.environ.setdefault("PULMONARY_IN_PROCESS", "1")
os.environ.setdefault(
    "MODEL_BUNDLE_URL",
    "https://github.com/Nadeeen232/Multimodal-Pulmonary-Diagnostics"
    "/releases/download/model-bundle-0.2.0-team29/SIC_0.2.0-team29_model_bundle.zip",
)
# SHA-256 of dist/SIC_0.2.0-team29_model_bundle.zip in docs/submission_manifest.json.
os.environ.setdefault(
    "MODEL_BUNDLE_SHA256", "839d4e39ee14dc96c7353fec4eff1e26d0c8f5f06da762ce95a229ec59bfc549"
)
for name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(name, "2")

runpy.run_path(str(ROOT / "ui/app.py"), run_name="__main__")
