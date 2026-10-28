"""Start and verify the trained local API/UI using one public validation case.

Services stay running after successful verification. Ports must be free. Only
processes created by this script are terminated if verification fails.
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import requests

from pulmonary.common import load_splits, read_config, save_json
from pulmonary.labels import clean_text

ROOT = Path(__file__).resolve().parents[1]


def wait_for(url, process, attempts=60):
    for _ in range(attempts):
        if process.poll() is not None:
            raise RuntimeError(f"Service exited with code {process.returncode}: {url}")
        try:
            response = requests.get(url, timeout=2)
            if response.ok:
                return response
        except requests.RequestException:
            pass
        time.sleep(1)
    raise RuntimeError(f"Service did not become ready: {url}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/local.yaml")
    args = parser.parse_args()
    os.chdir(ROOT)
    config = read_config(args.config)
    bundle = (Path(config["output_root"]) / "bundle").resolve()
    manifest = json.loads((bundle / "manifest.json").read_text())
    for port in [8080, 8501]:
        with socket.socket() as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError as exc:
                raise RuntimeError(
                    f"Port {port} is in use; existing services were left untouched"
                ) from exc
    folder = Path(config["reports_root"]) / "functional"
    folder.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, "MODEL_BUNDLE": str(bundle), "API_URL": "http://127.0.0.1:8080"}
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    processes = []
    logs = []
    try:
        for name, arguments in [
            ("api", ["-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8080"]),
            (
                "ui",
                [
                    "-m",
                    "streamlit",
                    "run",
                    "ui/app.py",
                    "--server.address=127.0.0.1",
                    "--server.port=8501",
                    "--server.headless=true",
                ],
            ),
        ]:
            handle = (folder / f"{name}.log").open("w", encoding="utf-8")
            logs.append(handle)
            processes.append(
                subprocess.Popen(
                    [sys.executable, *arguments],
                    cwd=ROOT,
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    creationflags=creationflags,
                )
            )
        health = wait_for("http://127.0.0.1:8080/ready", processes[0]).json()
        wait_for("http://127.0.0.1:8501/_stcore/health", processes[1])
        if health["model_version"] != manifest["model_version"]:
            raise RuntimeError("The API model version differs from the frozen selection")
        frame = load_splits(config)["validation"]
        row = frame.loc[frame.indication.map(clean_text).ne("")].iloc[0]
        # A canonical cached PNG is small and uses the same deterministic transform.
        image = (Path(config["output_root"]) / row.processed_image_path).read_bytes()
        results = {
            "status": "passed",
            "model_version": manifest["model_version"],
            "uid": str(row.uid),
            "split": "validation",
            "health": health,
            "services": {"api_pid": processes[0].pid, "ui_pid": processes[1].pid},
            "checks": {},
        }
        for route in ["predict", "explain"]:
            started = time.perf_counter()
            response = requests.post(
                f"http://127.0.0.1:8080/{route}",
                files={"image": ("validation.png", image, "image/png")},
                data={"indication": row.indication, "comparison": row.get("comparison", "")},
                timeout=120,
            )
            response.raise_for_status()
            result = response.json()
            prediction = result.get("prediction", result)
            if prediction["model_version"] != manifest["model_version"]:
                raise RuntimeError("Inference returned an unexpected model version")
            results["checks"][route] = {
                "status_code": response.status_code,
                "latency_seconds": time.perf_counter() - started,
            }
            save_json(folder / f"{route}_response.json", result)
        missing = requests.post(
            "http://127.0.0.1:8080/predict", data={"indication": "cough"}, timeout=10
        )
        if missing.status_code != 422:
            raise RuntimeError("Missing image was not rejected")
        results["checks"]["missing_image"] = {"status_code": missing.status_code}
        results["checks"]["ui_health"] = {"status_code": 200}
        save_json(folder / "local_stack_verification.json", results)
        print(json.dumps(results, indent=2))
        print("API: http://127.0.0.1:8080/docs\nUI: http://127.0.0.1:8501", flush=True)
    except Exception:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        raise
    finally:
        for handle in logs:
            handle.close()


if __name__ == "__main__":
    main()
