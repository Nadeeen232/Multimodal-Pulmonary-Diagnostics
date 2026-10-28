"""Export the implementation architecture as a presentation-ready PNG."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def render(output="docs/architecture.png"):
    fig, axis = plt.subplots(figsize=(14, 6.5), facecolor="#f8fafc")
    axis.set(xlim=(0, 14), ylim=(0, 6.5))
    axis.axis("off")
    axis.text(0.4, 6.0, "Pulmonary Review Assistant", fontsize=23, weight="bold", color="#0f172a")
    axis.text(
        0.4,
        5.55,
        "Reproducible training and interpretable multimodal inference",
        fontsize=13,
        color="#475569",
    )
    boxes = [
        (0.4, 3.5, "Indiana dataset", "Radiographs + indications"),
        (3.8, 3.5, "Pairing + transforms", "Grouped train / val / test"),
        (7.2, 3.5, "Tracked comparisons", "Text / ResNet / DenseNet / fusion"),
        (10.6, 3.5, "Validation + test", "Freeze choices; held-out evaluation"),
        (7.2, 1.2, "Frozen bundle", "Checksums + model version"),
        (3.8, 1.2, "FastAPI", "Prediction / XAI / written report"),
        (0.4, 1.2, "Streamlit", "Upload + review evidence"),
    ]
    for x, y, title, subtitle in boxes:
        axis.add_patch(
            FancyBboxPatch(
                (x, y),
                2.9,
                1.15,
                boxstyle="round,pad=0.05,rounding_size=0.12",
                facecolor="white",
                edgecolor="#cbd5e1",
            )
        )
        axis.text(
            x + 1.45, y + 0.73, title, ha="center", fontsize=13, weight="bold", color="#0f172a"
        )
        axis.text(x + 1.45, y + 0.32, subtitle, ha="center", fontsize=10, color="#475569")
    for start, end in [
        ((3.3, 4.08), (3.8, 4.08)),
        ((6.7, 4.08), (7.2, 4.08)),
        ((10.1, 4.08), (10.6, 4.08)),
        ((7.2, 1.78), (6.7, 1.78)),
        ((3.8, 1.78), (3.3, 1.78)),
    ]:
        axis.annotate(
            "", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": "#64748b", "lw": 1.7}
        )
    axis.plot([12.05, 12.05, 8.65], [3.5, 2.7, 2.7], color="#64748b", linewidth=1.7)
    axis.annotate(
        "",
        xy=(8.65, 2.35),
        xytext=(8.65, 2.7),
        arrowprops={"arrowstyle": "->", "color": "#64748b", "lw": 1.7},
    )
    axis.text(
        0.4,
        0.35,
        "Weak labels • no clinical validation • lower scores do not rule out abnormalities",
        fontsize=11,
        color="#64748b",
    )
    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor=fig.get_facecolor())
    fig.savefig(Path(output).with_suffix(".svg"), facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    render()
