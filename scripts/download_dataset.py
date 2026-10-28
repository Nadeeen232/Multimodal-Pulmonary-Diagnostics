"""Download Indiana with resumable KaggleHub caching and durable progress files."""

import contextlib
import json
import logging
import os
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    os.chdir(ROOT)
    os.environ.setdefault("KAGGLEHUB_CACHE", str(ROOT / "data/.cache"))
    os.environ.setdefault("TQDM_DISABLE", "1")
    data = ROOT / "data"
    data.mkdir(exist_ok=True)
    stop = threading.Event()
    state = {
        "status": "downloading",
        "source": "raddar/chest-xrays-indiana-university",
        "declared_dataset_bytes": 14134513955,
    }
    state_lock = threading.Lock()

    def save():
        with state_lock:
            archives = list(
                (data / ".cache/datasets/raddar/chest-xrays-indiana-university").glob("*.archive")
            )
            state["archive_bytes"] = sum(path.stat().st_size for path in archives if path.exists())
            state["updated_unix"] = time.time()
            temporary = data / "download_status.tmp"
            temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
            temporary.replace(data / "download_status.json")

    def monitor():
        while not stop.wait(10):
            save()

    save()
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    print("Indiana download running. Progress: data/download_status.json", flush=True)
    try:
        with (ROOT / "reports/dataset_download.log").open("a", encoding="utf-8") as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                logging.basicConfig(stream=log, level=logging.INFO)
                import kagglehub

                path = kagglehub.dataset_download("raddar/chest-xrays-indiana-university")
        manifest = {
            "source": "https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university",
            "downloaded_path": str(path),
        }
        (data / "download_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        state.update(status="complete", downloaded_path=str(path))
    except Exception as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        stop.set()
        thread.join(timeout=2)
        save()
    print(f"Indiana dataset ready: {path}", flush=True)


if __name__ == "__main__":
    main()
