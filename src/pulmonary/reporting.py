"""Human-readable conclusions derived from saved evidence, without invented gains."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd

from pulmonary.common import revision, save_json
from pulmonary.priority import get_priority


def markdown_table(frame, columns):
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        lines.append(
            "| "
            + " | ".join(
                f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c]) for c in columns
            )
            + " |"
        )
    return lines


def report(config):
    folder = Path(config["reports_root"])
    root = Path(config["output_root"])
    final = folder / "final/ablation_results.csv"
    lines = ["# Experiment evidence", "", f"Code revision: `{revision()}`", ""]
    columns = [
        "model",
        "family",
        "roc_auc",
        "pr_auc_average_precision",
        "precision",
        "recall",
        "f1",
        "threshold",
    ]
    if final.exists():
        table = pd.read_csv(final)
        lines += [
            "Primary metrics are study-level, averaging image probabilities within UID. PR-AUC is average precision.",
            "",
        ]
        lines += markdown_table(table, columns)
        manifest = json.loads((root / "bundle/manifest.json").read_text())
        selected = manifest["selected_model"]
        chosen = table.set_index("model").loc[selected]
        lines += [
            "",
            f"Serving model selected before test: **{selected}**, version `{manifest['model_version']}`. Selection rule: {manifest['selection']}",
        ]
        for name in ["text", "image"]:
            registry = json.loads((root / f"models/{name}_registry.json").read_text())
            reference = registry["best"]
            delta = chosen.roc_auc - table.set_index("model").loc[reference].roc_auc
            lines += [
                "",
                f"Observed test ROC-AUC difference against validation-selected {name} reference `{reference}`: {delta:+.4f}. This is a point estimate without external validation or a significance claim.",
            ]
        predictions = pd.read_csv(folder / f"final/{selected}/test_predictions.csv")
        predictions["priority_tier"] = [
            get_priority(p, manifest["threshold"])[0] for p in predictions.probability
        ]
        predictions.to_csv(folder / "final/review_priority_studies.csv", index=False)
        counts = predictions.priority_tier.value_counts().reindex(
            ["lower_review", "review", "higher_review"], fill_value=0
        )
        axis = counts.plot.bar(
            rot=15, figsize=(7, 4), title="Illustrative review ordering: held-out studies"
        )
        axis.set_ylabel("Studies")
        axis.figure.tight_layout()
        axis.figure.savefig(folder / "final/priority_tier_distribution.png", dpi=140)
        import matplotlib.pyplot as plt

        plt.close(axis.figure)
        costs = json.loads((folder / selected / "cost_references.json").read_text())
        save_json(
            folder / "final/priority_assumptions.json",
            {
                "fn_cost": manifest["fn_cost"],
                "fp_cost": manifest["fp_cost"],
                "calibrated": False,
                "priority_counts": {str(k): int(v) for k, v in counts.items()},
                "validation_cost_references": costs,
            },
        )
        policy = [
            "# Model selection and review policy",
            "",
            f"Selected model: `{selected}`. Threshold: {manifest['threshold']:.6f}. Both were fixed before test evaluation.",
            "",
            f"Assumed FN:FP costs: {manifest['fn_cost']}:{manifest['fp_cost']}. Minimum validation cost: {costs['minimum_validation_cost']:.1f}; always-review cost: {costs['always_review_cost']:.1f}; threshold-0.5 cost: {costs['threshold_0_5_cost']:.1f}.",
            "",
            "These are scenario calculations against weak labels, not observed clinical costs or realized benefits. Priority categories are illustrative; a lower score does not rule out an abnormality.",
        ]
        (folder / "best_model_selection.md").write_text("\n".join(policy), encoding="utf-8")
    else:
        lines += ["Final held-out evaluation has not been produced.", ""]
        registry_path = root / "models/text_registry.json"
        if registry_path.exists():
            registry = json.loads(registry_path.read_text())
            rows = [
                {"model": name, "family": item["family"], **item["validation"]}
                for name, item in registry["models"].items()
            ]
            lines += ["## Provisional validation text comparison", ""] + markdown_table(
                pd.DataFrame(rows), columns
            )
            lines += [
                "",
                "Image linkage may be unverified for metadata-only experiments. These are validation results, not test or multimodal results.",
            ]
    lines += [
        "",
        "## Interpretation",
        "",
        "Weak-label performance estimates agreement with a heuristic annotation mapping. It does not establish disease diagnosis, clinical benefit, or readiness for clinical deployment. FN:FP costs are illustrative assumptions. Repeated selection on a small validation set can overfit; external validation is needed. The implemented mapping includes some nonpulmonary thoracic findings and must be described precisely.",
    ]
    output = folder / "experiment_summary.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    return {"report": str(output)}
