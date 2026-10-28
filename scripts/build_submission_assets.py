"""Capture current-model explanation figures and operational evidence for submission."""

import base64
import csv
import json
import os
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests

from pulmonary.common import save_json

ROOT = Path(__file__).resolve().parents[1]


def build_xai_slide():
    output = ROOT / "reports/evidence/team29"
    positive = json.loads((output / "xai_uid_2348.json").read_text())
    false = json.loads((output / "xai_uid_2829.json").read_text())
    fig, axes = plt.subplots(
        1, 3, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 1, 1.25]}, layout="constrained"
    )
    for axis, uid, result, label in [
        (axes[0], "2348", positive, "Abnormal example"),
        (axes[1], "2829", false, "False positive (weak Normal)"),
    ]:
        axis.imshow(plt.imread(output / f"gradcam_uid_{uid}.png"))
        axis.axis("off")
        axis.set_title(
            f"{label}\nUID {uid} · score {result['prediction']['probability']:.1%}", fontsize=13
        )
    tokens = [t for t in positive["text_highlights"] if t["token"].lower() != "xxxx"][:6]
    values = [t["contribution"] for t in tokens]
    axes[2].barh(
        [t["token"] for t in tokens],
        values,
        color=["#c76a10" if v > 0 else "#1765b4" for v in values],
    )
    axes[2].invert_yaxis()
    axes[2].axvline(0, color="#64748b", lw=1)
    axes[2].set_title("Token occlusion · UID 2348", fontsize=13)
    axes[2].set_xlabel("Change in final score")
    axes[2].tick_params(axis="y", labelsize=13)
    fig.suptitle(
        "Actual deployed-model attribution · attention does not establish pathology", fontsize=14
    )
    fig.savefig(output / "xai_presentation.png", dpi=170)
    plt.close(fig)


def main():
    output = ROOT / "reports/evidence/team29"
    local = ROOT / "reports/generated/submission"
    output.mkdir(parents=True, exist_ok=True)
    local.mkdir(parents=True, exist_ok=True)
    api = "http://127.0.0.1:8080"
    response = requests.get(api + "/model-info", timeout=30)
    response.raise_for_status()
    info = response.json()
    assert info["model_version"] == "0.2.0-team29-47e281fc"
    save_json(output / "model_info.json", info)
    cases = list(csv.DictReader((ROOT / "demo_images/demo_cases.csv").open(encoding="utf-8")))
    for case in cases:
        with (ROOT / "demo_images" / case["filename"]).open("rb") as image:
            response = requests.post(
                api + "/explain",
                files={"image": ("xray.png", image, "image/png")},
                data={"indication": case["indication"], "comparison": case["comparison"]},
                timeout=120,
            )
        response.raise_for_status()
        result = response.json()
        assert result["prediction"]["model_version"] == info["model_version"]
        save_json(local / f"explanation_{case['uid']}.json", result)
        heatmap = output / f"gradcam_uid_{case['uid']}.png"
        heatmap.write_bytes(base64.b64decode(result["gradcam_base64"]))
        tokens = [t for t in result["text_highlights"] if t["token"].lower() != "xxxx"][:6]
        fig, axes = plt.subplots(
            1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [1, 1.3]}, layout="constrained"
        )
        axes[0].imshow(plt.imread(heatmap))
        axes[0].axis("off")
        axes[0].set_title("Grad-CAM — ResNet50 branch", fontsize=13)
        labels = [t["token"] for t in tokens]
        values = [t["contribution"] for t in tokens]
        axes[1].barh(labels, values, color=["#c76a10" if v > 0 else "#1765b4" for v in values])
        axes[1].invert_yaxis()
        axes[1].axvline(0, color="#64748b", lw=1)
        axes[1].set_title("Text attribution — token occlusion", fontsize=13)
        axes[1].set_xlabel("Change in final score when token is present")
        axes[1].tick_params(axis="y", labelsize=12)
        fig.suptitle(
            f"UID {case['uid']} · {result['predicted_class']} · score {result['prediction']['probability']:.1%}\n{info['model_version']} · attention and associations, not confirmed findings",
            fontsize=13,
        )
        fig.savefig(output / f"xai_uid_{case['uid']}.png", dpi=170)
        plt.close(fig)
        save_json(
            output / f"xai_uid_{case['uid']}.json",
            {k: v for k, v in result.items() if k != "gradcam_base64"},
        )
        print("Saved current explanation", case["uid"], flush=True)
    build_xai_slide()


if __name__ == "__main__":
    main()
