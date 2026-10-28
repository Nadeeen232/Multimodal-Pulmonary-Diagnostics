"""After real training succeeds, export the deck and verify the live local stack."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    os.chdir(ROOT)
    folder = ROOT / "reports/generated/finalization"
    folder.mkdir(parents=True, exist_ok=True)
    status_path = folder / "status.json"
    state = {"status": "waiting_for_real_pipeline", "stages": []}

    def save():
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
        temporary.replace(status_path)

    save()
    pipeline = ROOT / "reports/generated/pipeline/status.json"
    while True:
        try:
            current = json.loads(pipeline.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(5)
            continue
        if current["status"] in {"failed", "download_failed"}:
            state.update(status="blocked_by_pipeline_failure", pipeline=current)
            save()
            raise SystemExit(1)
        if current["status"] == "complete":
            break
        time.sleep(10)
    # The pipeline is finished; refresh its old download snapshot for clarity.
    current["download"] = json.loads((ROOT / "data/download_status.json").read_text())
    pipeline.write_text(json.dumps(current, indent=2), encoding="utf-8")
    for script in ["build_presentation.py", "verify_local_stack.py"]:
        state.update(status="running", current_stage=script)
        save()
        with (folder / f"{Path(script).stem}.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / script), "--config", "configs/local.yaml"],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        state["stages"].append({"script": script, "returncode": result.returncode})
        if result.returncode:
            state["status"] = "failed"
            save()
            raise SystemExit(result.returncode)
    state.update(
        status="complete",
        docker="not verified: Docker is unavailable on this machine",
        ui_url="http://127.0.0.1:8501",
        api_docs_url="http://127.0.0.1:8080/docs",
    )
    state.pop("current_stage", None)
    save()
    (ROOT / "reports/generated/PROJECT_STATUS.md").write_text(
        "# Real-data run and local demonstration\n\nThe real pipeline and local stack verification completed. See `experiment_summary.md`, `final/ablation_results.csv`, `presentation.pptx`, and `functional/local_stack_verification.json` for actual evidence.\n\n"
        "UI: http://127.0.0.1:8501\n\nAPI docs: http://127.0.0.1:8080/docs\n\n"
        "Container build/run remains unverified because Docker is unavailable. Assigned team responsibilities are recorded; each member still needs to review and confirm their own contributions. Outputs remain a research prototype against weak annotation labels.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
