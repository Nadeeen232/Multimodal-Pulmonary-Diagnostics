"""Export existing real experiment evidence; never train or reevaluate the test set."""

import json
import shutil
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports/generated"
OUTPUT = ROOT / "reports/evidence"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    copies = {
        "final/ablation_results.csv": "ablation_results.csv",
        "final/ablation.png": "ablation.png",
        "fusion_early/threshold_cost_curve.png": "threshold_cost_curve.png",
        "final/fusion_early/test_raw_reliability.png": "test_raw_reliability.png",
        "text/optuna_trials.csv": "optuna_trials.csv",
        "text/optuna_best_params.json": "optuna_best_params.json",
        "functional/docker_stack_verification.json": "docker_stack_verification.json",
    }
    models = ["text_logistic", "resnet50", "densenet121", "fusion_early", "fusion_late"]
    for model in models:
        copies[f"final/{model}/test_curves.png"] = f"{model}_test_curves.png"
        copies[f"final/{model}/test_metrics.json"] = f"{model}_test_metrics.json"
    for source, destination in copies.items():
        shutil.copy2(REPORTS / source, OUTPUT / destination)
    con = sqlite3.connect(f"file:{(ROOT / 'artifacts/mlflow.db').as_posix()}?mode=ro", uri=True)
    runs = pd.read_sql_query(
        "SELECT r.run_uuid,r.name,r.status,e.name AS experiment FROM runs r "
        "JOIN experiments e ON r.experiment_id=e.experiment_id", con,
    )
    summary = runs.groupby(["experiment", "status"]).size().reset_index(name="runs")
    (OUTPUT / "tracking_snapshot.json").write_text(json.dumps({
        "run_count": len(runs), "experiments": summary.to_dict("records"),
        "runs": runs.to_dict("records"),
        "source_revision": json.loads((ROOT / "artifacts/bundle/manifest.json").read_text())["revision"],
    }, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    histories = []
    for model, ax in zip(["resnet50", "densenet121", "fusion_early"], axes, strict=True):
        run = runs.loc[(runs.name == model) & runs.experiment.ne("sic-final-evaluation")].iloc[0]
        values = pd.read_sql_query(
            "SELECT key,value,step FROM metrics WHERE run_uuid=? "
            "AND key IN ('train_loss','validation_loss') ORDER BY step", con,
            params=(run.run_uuid,),
        )
        pivot = values.pivot(index="step", columns="key", values="value")
        pivot.to_csv(OUTPUT / f"{model}_learning_history.csv")
        for key, label in [("train_loss", "Train"), ("validation_loss", "Validation")]:
            ax.plot(pivot.index + 1, pivot[key], marker="o", markersize=3, label=label)
        best = int(pivot.validation_loss.idxmin()) + 1
        ax.axvline(best, color="#64748b", linestyle="--", alpha=0.7, label=f"Best epoch {best}")
        ax.set(title=model, xlabel="Epoch (1-based)", ylabel="Weighted cross-entropy loss")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
        histories.append({"model": model, "epochs_completed": len(pivot), "best_epoch": best,
                          "best_validation_loss": float(pivot.validation_loss.min())})
    fig.suptitle("Saved MLflow training histories — early stopping, patience 3")
    fig.tight_layout()
    fig.savefig(OUTPUT / "learning_curves.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    con.close()
    (OUTPUT / "training_summary.json").write_text(json.dumps(histories, indent=2))
    trials = pd.read_csv(OUTPUT / "optuna_trials.csv")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.scatter(trials.number + 1, trials.value, color="#64748b", label="Trial grouped-CV ROC-AUC")
    ax.plot(trials.number + 1, trials.value.cummax(), color="#193eb0", label="Best so far")
    ax.set(title="Logistic regression: 20 completed Optuna trials",
           xlabel="Trial (1-based)", ylabel="Training grouped-CV ROC-AUC")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.savefig(OUTPUT / "optuna_search.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    ablation = pd.read_csv(OUTPUT / "ablation_results.csv")
    selected = ablation.set_index("model").loc[models]
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(models, selected.accuracy, color=["#94a3b8", "#193eb0", "#4866c7", "#0f766e", "#14b8a6"])
    ax.bar_label(bars, labels=[f"{v:.2%}" for v in selected.accuracy], padding=4)
    ax.set(title="Final held-out accuracy — validation-selected thresholds", ylabel="Accuracy", ylim=(0, 0.85))
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(OUTPUT / "held_out_accuracy.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    cases = [json.loads(path.read_text()) | {"path": path}
             for path in sorted((REPORTS / "xai_examples").glob("case_*.json"))]
    records = []
    for category in ["TN", "TP", "FP", "FN"]:
        record = next(row for row in cases if row["selection_category"] == category)
        shutil.copy2(record["path"].with_suffix(".png"), OUTPUT / f"xai_{category.lower()}.png")
        shutil.copy2(record["path"], OUTPUT / f"xai_{category.lower()}.json")
        records.append({"category": category, "uid": record["uid"],
                        "score": record["prediction"]["probability"]})
    record = next(row for row in cases if row["selection_category"] == "TN")
    tokens = record["text_highlights"][:10][::-1]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(np.arange(len(tokens)), [t["contribution"] for t in tokens],
            color=["#dc2626" if t["contribution"] > 0 else "#2563eb" for t in tokens])
    ax.set_yticks(np.arange(len(tokens)), [t["token"] for t in tokens])
    ax.axvline(0, color="#64748b", linewidth=1)
    ax.set(title=f"Token occlusion — validation UID {record['uid']} (not SHAP)",
           xlabel="Full score − score after removing token")
    fig.savefig(OUTPUT / "text_token_occlusion.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    (OUTPUT / "xai_selection.json").write_text(json.dumps(records, indent=2))
    print(json.dumps({"training": histories, "mlflow_runs": len(runs),
                      "trials": len(trials), "xai_cases": records}, indent=2))


if __name__ == "__main__":
    main()
