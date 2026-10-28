"""Study-level evaluation, validation-only threshold choice, and evidence plots."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from pulmonary.common import save_json


def study_predictions(frame, probabilities):
    result = frame[["uid", "target"]].copy()
    result["probability"] = np.asarray(probabilities, dtype=float)
    if len(result) == 0 or not np.isfinite(result.probability).all():
        raise ValueError("Predictions are empty or nonfinite")
    if not result.probability.between(0, 1).all():
        raise ValueError("Probabilities must lie in [0, 1]")
    if result.groupby("uid").target.nunique().max() != 1:
        raise ValueError("A study has inconsistent labels")
    return (
        result.groupby("uid", sort=True)
        .agg(target=("target", "first"), probability=("probability", "mean"))
        .reset_index()
    )


def threshold_table(frame, probabilities, fn_cost=10.0, fp_cost=1.0):
    studies = study_predictions(frame, probabilities)
    y, p = studies.target.to_numpy(), studies.probability.to_numpy()
    thresholds = np.unique(np.r_[0.0, p, np.nextafter(p, 1.0), 0.5, 1.0])
    rows = []
    for threshold in thresholds:
        tn, fp, fn, tp = confusion_matrix(y, p >= threshold, labels=[0, 1]).ravel()
        rows.append(
            {
                "threshold": float(threshold),
                "fn": int(fn),
                "fp": int(fp),
                "cost": float(fn_cost * fn + fp_cost * fp),
                "f1": float(f1_score(y, p >= threshold, zero_division=0)),
            }
        )
    return pd.DataFrame(rows)


def choose_threshold(frame, probabilities, config):
    table = threshold_table(frame, probabilities, config["fn_cost"], config["fp_cost"])
    # Fixed tie-break: best F1, then higher threshold, among minimum-cost candidates.
    row = table.sort_values(["cost", "f1", "threshold"], ascending=[True, False, False]).iloc[0]
    return float(row.threshold), table


def cost_evidence(table, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    table.to_csv(folder / "threshold_costs.csv", index=False)
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.plot(table.threshold, table.cost)
    axis.set(
        xlabel="Decision threshold",
        ylabel="Assumed total error cost",
        title="Validation cost sensitivity",
    )
    fig.tight_layout()
    fig.savefig(folder / "threshold_cost_curve.png", dpi=140)
    plt.close(fig)
    always_review = table.loc[table.threshold.eq(0)].iloc[0]
    default = table.loc[table.threshold.eq(0.5)].iloc[0]
    save_json(
        folder / "cost_references.json",
        {
            "always_review_cost": float(always_review.cost),
            "threshold_0_5_cost": float(default.cost),
            "minimum_validation_cost": float(table.cost.min()),
            "interpretation": "Costs are scenario assumptions, not measured clinical or financial losses. Compare against reviewing every study before claiming review savings.",
        },
    )


def metrics(frame, probabilities, threshold):
    studies = study_predictions(frame, probabilities)
    y, p = studies.target.to_numpy(), studies.probability.to_numpy()
    pred = p >= threshold
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc_average_precision": float(average_precision_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "threshold": float(threshold),
        "studies": len(studies),
        "image_rows": len(frame),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


def evidence(frame, probabilities, threshold, folder, prefix):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    studies = study_predictions(frame, probabilities)
    studies.to_csv(folder / f"{prefix}_predictions.csv", index=False)
    result = metrics(frame, probabilities, threshold)
    save_json(folder / f"{prefix}_metrics.json", result)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    RocCurveDisplay.from_predictions(studies.target, studies.probability, ax=axes[0])
    PrecisionRecallDisplay.from_predictions(studies.target, studies.probability, ax=axes[1])
    ConfusionMatrixDisplay.from_predictions(
        studies.target,
        studies.probability >= threshold,
        labels=[0, 1],
        display_labels=["Proxy negative", "Proxy positive"],
        ax=axes[2],
        colorbar=False,
    )
    fig.tight_layout()
    fig.savefig(folder / f"{prefix}_curves.png", dpi=140)
    plt.close(fig)
    from sklearn.calibration import CalibrationDisplay

    fig, axis = plt.subplots(figsize=(5, 4))
    CalibrationDisplay.from_predictions(
        studies.target, studies.probability, n_bins=8, strategy="quantile", ax=axis
    )
    axis.set_title("Raw score reliability against weak labels")
    fig.tight_layout()
    fig.savefig(folder / f"{prefix}_raw_reliability.png", dpi=140)
    plt.close(fig)
    return result
