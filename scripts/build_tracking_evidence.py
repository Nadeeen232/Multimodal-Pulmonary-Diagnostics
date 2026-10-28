"""Export read-only MLflow evidence and draw a summary; never creates tracking runs."""

import json
import sqlite3
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "reports/evidence"
SNAPSHOT = EVIDENCE / "team29/mlflow_tracking_snapshot.json"
NAMES = {
    "text_only": "Text-only",
    "resnet50": "ResNet50",
    "multimodal_fusion": "Multimodal Fusion",
}


def export_snapshot():
    database = ROOT / "artifacts/team_classification/mlflow.db"
    if not database.exists():
        if SNAPSHOT.exists():
            return json.loads(SNAPSHOT.read_text(encoding="utf-8"))
        raise FileNotFoundError("Restore the tracking archive or the public snapshot first.")
    # Read-only SQLite access avoids MLflow migrations, new runs or URI rewrites.
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        runs = []
        for run_id, name, experiment, status in connection.execute(
            "SELECT r.run_uuid,r.name,e.name,r.status FROM runs r "
            "JOIN experiments e ON r.experiment_id=e.experiment_id ORDER BY r.start_time"
        ):
            params = dict(
                connection.execute("SELECT key,value FROM params WHERE run_uuid=?", (run_id,))
            )
            metrics = dict(
                connection.execute(
                    "SELECT key,value FROM latest_metrics WHERE run_uuid=?", (run_id,)
                )
            )
            runs.append(
                dict(
                    run_id=run_id,
                    name=name,
                    experiment=experiment,
                    status=status,
                    params=params,
                    metrics=metrics,
                )
            )
    snapshot = {
        "source": "artifacts/team_classification/mlflow.db",
        "method": "Read-only SQLite export; artifact URIs and workstation paths excluded",
        "model_version": "0.2.0-team29-47e281fc",
        "run_count": len(runs),
        "statuses": dict(Counter(r["status"] for r in runs)),
        "runs": runs,
        "text_selection": json.loads(
            (ROOT / "artifacts/team_classification/text_selection.json").read_text()
        ),
        "fusion_selection": json.loads(
            (ROOT / "artifacts/team_classification/selection.json").read_text()
        ),
    }
    SNAPSHOT.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    return snapshot


def tracking_markdown():
    current = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    historical = json.loads((EVIDENCE / "tracking_snapshot.json").read_text())
    trials = pd.read_csv(EVIDENCE / "optuna_trials.csv")
    complete = trials.loc[trials.state.eq("COMPLETE")].sort_values("number")
    best = complete.loc[complete.value.idxmax()]
    params = json.loads((EVIDENCE / "optuna_best_params.json").read_text())
    rows = []
    for model, label in NAMES.items():
        run = next(r for r in current["runs"] if r["name"] == model + "-frozen-evaluation")
        m = run["metrics"]
        rows.append(
            f"| {label} | `{run['run_id'][:8]}` | {100 * m['test_accuracy']:.2f}% | {m['test_f1']:.4f} | {m['test_roc_auc']:.4f} |"
        )
    checkpoints = sorted(set([int(complete.iloc[0].number), 4, 9, int(complete.iloc[-1].number)]))
    history = []
    for number in checkpoints:
        observed = complete.loc[complete.number.le(number)]
        history.append(f"| {number} | {observed.iloc[-1].value:.4f} | {observed.value.max():.4f} |")
    return f"""### MLflow experiment tracking and Optuna optimization history

The current database contains **{current["run_count"]} runs** in
`team29-classification-reproduction`: {current["statuses"].get("FINISHED", 0)} FINISHED,
{current["statuses"].get("KILLED", 0)} KILLED and {current["statuses"].get("FAILED", 0)} FAILED.
Five finished runs log frozen test evaluations; the other two finished runs are
CNN training. Interrupted attempts remain visible rather than being discarded.
MLflow records parameters, epoch training/validation losses, final test metrics,
selection manifests, checkpoint hashes and saved plots. Training runs record the
seed and source digest; final evaluation runs link the cohort/selection manifests.

| Model family / representative | MLflow run prefix | Test accuracy | Test F1 | Test ROC-AUC |
| --- | --- | ---: | ---: | ---: |
{chr(10).join(rows)}

These metrics are read directly from the three frozen-evaluation MLflow runs,
covering text-only, image-only and multimodal families on the same 356 test studies.
Full run IDs, statuses, parameters and metrics are in
`reports/evidence/team29/mlflow_tracking_snapshot.json`.

![MLflow and Optuna evidence summary — generated from saved records, not a dashboard screenshot](../reports/evidence/mlflow_dashboard.png)

**Current validation search.** The text model compares 12 settings (six C values,
two class-weight settings). Its selected C is **0.3**, class weighting is **None**,
solver is **liblinear**, and validation ROC-AUC is **{current["text_selection"]["validation_auc"]:.4f}**.
Fusion compares 21 image weights using validation ROC-AUC and selects **70% image /
30% text**. Balanced-accuracy threshold selection on validation gives **0.49** for
fusion. CNN early stopping uses patience 2; retained checkpoints are ResNet50
epoch 7 and DenseNet121 epoch 10. This current search is not an Optuna study.

**Historical Optuna study — separate cohort and feature pipeline.**
`tracking_snapshot.json` records **{historical["run_count"]} historical MLflow runs**.
`optuna_trials.csv` records **{len(complete)} completed trials**, numbered 0–19,
using seeded TPE to maximize mean grouped training-fold ROC-AUC. TF-IDF is refitted
within each fold. Search bounds are C in [0.001, 100] (log scale) and max_features
in {{500, 1000, 2000}}. The best trial is **{int(best.number)}**, with mean CV ROC-AUC
**{best.value:.4f}**, C = **{params["C"]:.10f}** and max_features = **{params["max_features"]}**.
The classifier uses balanced class weights. These are historical development
results, not the deployed text model's settings or new test-set tuning.

| Historical trial (zero-based) | Trial mean CV ROC-AUC | Best observed through this trial |
| --- | ---: | ---: |
{chr(10).join(history)}

The chart includes every trial and its cumulative best; the full CSV preserves
parameters, states, timestamps and duration. Current threshold and fusion choices
were frozen on validation before final test evaluation. No test metric is used
as a tuning objective.

**Inspect the local MLflow dashboard.** Run from the project root:

```powershell
.\\.venv\\Scripts\\python.exe -m mlflow ui --backend-store-uri sqlite:///artifacts/team_classification/mlflow.db --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000` and select `team29-classification-reproduction`.
Compare the five `*-frozen-evaluation` runs using `test_accuracy`, `test_f1` and
`test_roc_auc`; inspect CNN runs for loss histories. The historical database can
be viewed separately with `--backend-store-uri sqlite:///artifacts/mlflow.db`
on port 5001. Databases/artifact directories are excluded from Git; restore the
separate tracking archive before opening the current dashboard on a fresh checkout.
The summary chart remains viewable without either database.
"""


def main():
    snapshot = export_snapshot()
    trials = pd.read_csv(EVIDENCE / "optuna_trials.csv")
    trials = trials.loc[trials.state.eq("COMPLETE")].sort_values("number")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig = plt.figure(figsize=(14, 8), facecolor="white")
    grid = fig.add_gridspec(
        2, 2, top=0.80, bottom=0.12, left=0.07, right=0.96, hspace=0.55, wspace=0.35
    )
    fig.text(0.055, 0.94, "MLflow tracking & Optuna", fontsize=25, weight="bold", color="#193EB0")
    fig.text(
        0.055,
        0.89,
        "Evidence summary | current bundle 0.2.0-team29 | separate historical tuning",
        fontsize=12,
        color="#516278",
    )
    status = snapshot["statuses"]
    ax = fig.add_subplot(grid[0, 0])
    labels = ["FINISHED", "KILLED", "FAILED"]
    bars = ax.barh(
        labels, [status.get(x, 0) for x in labels], color=["#193EB0", "#E6A323", "#CD5B59"]
    )
    ax.bar_label(bars, padding=5)
    ax.set(
        title=f"Current MLflow: {snapshot['run_count']} real runs", xlabel="Run count", xlim=(0, 9)
    )
    ax = fig.add_subplot(grid[0, 1])
    frame = pd.read_csv(EVIDENCE / "team29/ablation_results.csv").set_index("model")
    scores = [frame.loc[m, "roc_auc"] for m in NAMES]
    bars = ax.barh(list(NAMES.values()), scores, color=["#87A4DF", "#4A72CE", "#193EB0"])
    ax.bar_label(bars, labels=[f"{x:.4f}" for x in scores], padding=5)
    ax.set(
        xlim=(0, 1),
        title="Current families: held-out ROC-AUC",
        xlabel="356 studies | frozen evaluation",
    )
    ax = fig.add_subplot(grid[1, :])
    ax.plot(
        trials.number, trials.value, "o-", color="#87A4DF", label="Trial mean grouped-CV ROC-AUC"
    )
    ax.step(
        trials.number,
        trials.value.cummax(),
        where="post",
        color="#193EB0",
        linewidth=2.2,
        label="Best observed",
    )
    best = trials.loc[trials.value.idxmax()]
    ax.scatter([best.number], [best.value], s=130, marker="*", color="#E6A323", zorder=5)
    ax.set(
        title=f"Historical v1 Optuna: {len(trials)} completed trials | best #{int(best.number)} = {best.value:.4f}",
        xlabel="Trial number (zero-based) — different cohort / text pipeline",
        ylabel="Mean CV ROC-AUC",
    )
    ax.set_xticks(range(0, len(trials), 2))
    ax.legend(loc="lower right", fontsize=10)
    fig.text(
        0.055,
        0.035,
        "Generated from MLflow records and optuna_trials.csv; this is a summary chart, not a UI screenshot.",
        fontsize=10,
        color="#516278",
    )
    output = EVIDENCE / "mlflow_dashboard.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)
    # A wider, simpler variant keeps chart labels readable below slide bullets.
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1, 1.8]})
    labels = ["Text", "Vision", "Fusion"]
    bars = axes[0].barh(labels, scores, color=["#87A4DF", "#4A72CE", "#193EB0"])
    axes[0].bar_label(bars, labels=[f"{x:.4f}" for x in scores], padding=4)
    axes[0].set(xlim=(0, 1.1), title="Current test ROC-AUC")
    axes[1].plot(trials.number, trials.value, "o-", color="#87A4DF", label="Trial")
    axes[1].step(trials.number, trials.value.cummax(), where="post", color="#193EB0", label="Best")
    axes[1].set(
        title="Historical Optuna: 20 trials", xlabel="Trial number", ylabel="Grouped-CV ROC-AUC"
    )
    axes[1].legend(loc="lower right", fontsize=9)
    fig.tight_layout(w_pad=2)
    fig.savefig(EVIDENCE / "mlflow_presentation.png", dpi=180)
    plt.close(fig)
    print(f"Saved {output.relative_to(ROOT)} and read-only tracking snapshot")


if __name__ == "__main__":
    main()
