"""Run the team's selected classification approach without a notebook."""

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["TORCH_CUDNN_V8_API_DISABLED"] = "1"
os.environ.setdefault(
    "TORCH_HOME", str(Path(__file__).resolve().parents[1] / "artifacts/torch-cache")
)
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import torch

from pulmonary.common import read_config, save_json
from pulmonary.team_classification import finalize, prepare, train_text, train_vision


class Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, value):
        self.terminal.write(value)
        self.log.write(value)
        self.flush()
        return len(value)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "text", "images", "finalize", "all", "train"])
    parser.add_argument("--config", default="configs/local.yaml")
    parser.add_argument("--output", default="artifacts/team_classification")
    args = parser.parse_args()
    torch.set_num_threads(2)
    config = read_config(args.config)
    output = Path(args.output)
    if args.stage in {"images", "all", "train"}:
        Path("reports").mkdir(exist_ok=True)
        sys.stdout = Tee(sys.stdout, Path("reports/team_training.log").open("a", encoding="utf-8"))
        sys.stderr = Tee(
            sys.stderr, Path("reports/team_training_error.log").open("a", encoding="utf-8")
        )
    state = {
        "pid": os.getpid(),
        "stage": args.stage,
        "started": datetime.now(timezone.utc).isoformat(),
        "status": "running",
    }
    save_json(output / "process_status.json", state)
    try:
        run(args, config, output)
    except BaseException as error:
        save_json(
            output / "process_status.json", {**state, "status": "failed", "error": str(error)}
        )
        raise
    save_json(output / "process_status.json", {**state, "status": "complete"})


def run(args, config, output):
    if args.stage in {"prepare", "all"}:
        prepare(config["data_root"], output)
    if args.stage in {"text", "all", "train"}:
        train_text(output)
    if args.stage in {"images", "all", "train"}:
        for name in ["resnet50", "densenet121"]:
            train_vision(output, name)
    if args.stage in {"finalize", "all", "train"}:
        finalize(output)


if __name__ == "__main__":
    main()
