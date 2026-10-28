"""Run the real file-based workflow, optionally after an existing download finishes.

Examples:
    python scripts/run_pipeline.py --wait-for-download
    python scripts/run_pipeline.py --download
Each stage gets a log and status record. Failed stages halt the workflow.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-for-download", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--config", default="configs/local.yaml")
    args = parser.parse_args()
    os.chdir(ROOT)
    folder = ROOT / "reports/generated/pipeline"
    folder.mkdir(parents=True, exist_ok=True)
    state_path = folder / "status.json"
    state = {"status": "starting", "stages": []}

    def save():
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def stage(command, config=None):
        state.update(status="running", current_stage=command)
        save()
        argv = [sys.executable, "-m", "pulmonary.cli"]
        if config:
            argv += ["--config", config]
        argv.append(command)
        with (folder / f"{command}.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=False)
        state["stages"].append({"command": command, "returncode": result.returncode})
        if result.returncode:
            state["status"] = "failed"
            save()
            raise SystemExit(result.returncode)
        save()

    if args.download:
        stage("download")
    if args.wait_for_download:
        state["status"] = "waiting_for_dataset"
        save()
        while not (ROOT / "data/download_manifest.json").exists():
            download_status = ROOT / "data/download_status.json"
            if download_status.exists():
                download = json.loads(download_status.read_text())
                state["download"] = download
                if download.get("status") == "failed":
                    state["status"] = "download_failed"
                    save()
                    raise SystemExit("Dataset download failed; see reports/dataset_download.log")
                save()
            time.sleep(10)
        manifest = json.loads((ROOT / "data/download_manifest.json").read_text())
        if not Path(manifest["downloaded_path"]).is_dir():
            raise FileNotFoundError("Download manifest does not point to extracted data")
        try:
            dataset_path = Path(manifest["downloaded_path"]).relative_to(ROOT).as_posix()
        except ValueError:
            dataset_path = manifest["downloaded_path"]
        (ROOT / args.config).write_text(
            yaml.safe_dump({"extends": "default.yaml", "data_root": dataset_path}), encoding="utf-8"
        )
    for command in [
        "prepare",
        "train-text",
        "train-images",
        "train-fusion",
        "export",
        "evaluate-test",
        "explain-examples",
        "report",
    ]:
        stage(command, args.config)
    state["status"] = "complete"
    state.pop("current_stage", None)
    save()


if __name__ == "__main__":
    main()
