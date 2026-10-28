"""Create plots and tracking evidence from frozen team-reproduction results."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd

from pulmonary.common import file_hash, save_json
from pulmonary.evaluation import evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="artifacts/team_classification")
    parser.add_argument("--output", default="reports/evidence/team29")
    args = parser.parse_args()
    source, output = Path(args.source), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(source / "ablation_results.csv")
    predictions = pd.read_csv(source / "study_test_predictions.csv", dtype={"uid": str})
    selection = json.loads((source / "selection.json").read_text())
    image_rows = int(pd.read_csv(source / "images.csv").split.eq("test").sum())
    for row in table.to_dict("records"):
        probability = predictions[f"p_{row['model']}"]
        result = evidence(predictions, probability, row["threshold"], output, row["model"])
        result.update(image_rows=image_rows, input_predictions="study-mean scores")
        save_json(output / f"{row['model']}_metrics.json", result)
    table.to_csv(output / "ablation_results.csv", index=False)
    for model in ["resnet50", "densenet121"]:
        history = pd.read_csv(source / f"{model}_history.csv")
        fig, axis = plt.subplots(figsize=(7, 4))
        axis.plot(history.epoch, history.train_loss, label="Training")
        axis.plot(history.epoch, history.val_loss, label="Validation")
        axis.axvline(3.5, color="gray", linestyle="--", label="Fine-tuning begins")
        axis.set(xlabel="Epoch", ylabel="Weighted cross-entropy", title=model)
        axis.legend()
        fig.tight_layout()
        fig.savefig(output / f"{model}_loss.png", dpi=150)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = table.model.str.replace("_", " ")
    for axis, metric in zip(axes, ["accuracy", "roc_auc"], strict=True):
        axis.barh(labels, table[metric])
        axis.set_xlim(0, 1)
        axis.set_title(metric.replace("_", " ").upper())
    fig.tight_layout()
    fig.savefig(output / "ablation.png", dpi=150)
    plt.close(fig)
    mlflow.set_tracking_uri("sqlite:///" + (source / "mlflow.db").resolve().as_posix())
    mlflow.set_experiment("team29-classification-reproduction")
    for model in table.model:
        row = table.loc[table.model.eq(model)].iloc[0]
        with mlflow.start_run(run_name=model + "-frozen-evaluation"):
            mlflow.log_params(
                {
                    "cohort_manifest": selection["cohort_manifest"],
                    "selection_objective": selection["objective"],
                    "threshold": row.threshold,
                    "image_weight": selection["image_weight"]
                    if model == "multimodal_fusion"
                    else 0,
                }
            )
            mlflow.log_metrics(
                {f"test_{key}": float(row[key]) for key in ["accuracy", "f1", "roc_auc"]}
            )
            mlflow.log_artifact(str(source / "selection.json"))
            mlflow.log_artifact(str(source / "cohort_manifest.json"))
            if model in {"resnet50", "densenet121"}:
                history = pd.read_csv(source / f"{model}_history.csv")
                selected_epoch = int(history.loc[history.val_loss.idxmin(), "epoch"])
                mlflow.log_params(
                    {
                        "selected_epoch": selected_epoch,
                        "checkpoint_sha256": file_hash(source / f"{model}.pt"),
                    }
                )
                mlflow.log_artifact(str(source / f"{model}_history.csv"))
            if model == "text_only":
                mlflow.log_artifact(str(source / "text_selection.json"))
                mlflow.log_artifact(str(source / "text_search.csv"))
            for path in output.glob(model + "*"):
                mlflow.log_artifact(str(path))
    manifest = {
        "source": "local team29 reproduction; not the PDF's supplied scores",
        "cohort_manifest": selection["cohort_manifest"],
        "files": {
            p.name: file_hash(p)
            for p in output.iterdir()
            if p.is_file() and p.name != "manifest.json"
        },
    }
    save_json(output / "manifest.json", manifest)
    print(table.to_string(index=False))
    print(f"Evidence saved to {output}")


if __name__ == "__main__":
    main()
