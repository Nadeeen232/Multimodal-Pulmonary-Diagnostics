"""Dataset summaries and training-only annotation review samples."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def write_eda(reports, train, audit, folder, seed):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    reports.medical_status.value_counts().reindex(["normal", "abnormal", "uncertain"]).plot.bar(
        ax=axes[0], rot=20
    )
    axes[0].set_title("Report-level weak labels")
    fields = [
        name for name in ["indication", "comparison", "findings", "impression"] if name in reports
    ]
    missing = pd.Series({name: int(reports[name].eq("").sum()) for name in fields})
    missing.plot.bar(ax=axes[1], rot=20)
    axes[1].set_title("Missing source text fields")
    unique_train = train.drop_duplicates("uid").copy()
    unique_train["text_word_count"] = unique_train.model_text.str.split().str.len()
    for target, group in unique_train.groupby("target"):
        axes[2].hist(group.text_word_count, bins=20, alpha=0.6, label=f"Proxy label {target}")
    axes[2].legend()
    axes[2].set(xlabel="Words", ylabel="Training studies", title="Training indication lengths")
    fig.tight_layout()
    fig.savefig(folder / "data_overview.png", dpi=140)
    plt.close(fig)
    unique_train.groupby("target").text_word_count.describe().to_csv(
        folder / "training_text_statistics.csv"
    )
    review = unique_train.groupby("target", group_keys=False).apply(
        lambda part: part.sample(min(20, len(part)), random_state=seed), include_groups=False
    )
    columns = [
        c
        for c in ["uid", "problems", "mesh", "indication", "medical_status", "target"]
        if c in review
    ]
    review[columns].to_csv(folder / "training_label_review_sample.csv", index=False)
    # Uncertain reports are never used as held-out binary labels.
    uncertain = reports.loc[reports.medical_status.eq("uncertain")]
    uncertain.sample(min(20, len(uncertain)), random_state=seed).to_csv(
        folder / "uncertain_label_review_sample.csv", index=False
    )
    (folder / "label_review_instructions.md").write_text(
        "# Annotation review\n\nSamples are training studies and excluded uncertain reports. Review whether structured tags support the intended proxy task; record disagreements and exclusions in a new versioned label specification. Do not change rules in response to test metrics.\n\n"
        f"Image verification: {audit['image_validation']}. Grouping: {audit['grouping_column']}.\n",
        encoding="utf-8",
    )
