"""Validation-based selection, checked serving bundles, and final test evidence."""

import json
import shutil
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import torch
from PIL import Image

from pulmonary.common import (
    device_for,
    file_hash,
    load_splits,
    require_verified_images,
    revision,
    save_json,
    source_digest,
    split_digest,
)
from pulmonary.evaluation import evidence
from pulmonary.image_models import predictions
from pulmonary.images import PREPROCESS_VERSION
from pulmonary.inference import InferenceEngine
from pulmonary.labels import LABEL_VERSION
from pulmonary.networks import FusionHead, load_encoder
from pulmonary.tracking import tracked


def read_registries(config):
    root = Path(config["output_root"]) / "models"
    result = {}
    for name in ["text", "image", "fusion"]:
        registry = json.loads((root / f"{name}_registry.json").read_text())
        if registry["split_manifest_sha256"] != split_digest(config):
            raise ValueError(f"{name} models use a different split manifest")
        result.update(registry["models"])
    return result


def export_bundle(config):
    require_verified_images(config)
    read_registries(config)
    root = Path(config["output_root"]) / "models"
    registry = json.loads((root / "fusion_registry.json").read_text())
    if registry["split_manifest_sha256"] != split_digest(config):
        raise ValueError("Fusion registry and current splits differ")
    selected = registry["models"][registry["best"]]
    folder = Path(config["output_root"]) / "bundle"
    if folder.exists():
        raise FileExistsError(
            "Bundle already exists. Use a new output_root for a new experiment/version."
        )
    folder.mkdir(parents=True)
    copies = {"backbone.pt": root / f"{selected['backbone']}.pt"}
    if selected["kind"] == "early":
        copies.update(
            {
                "fusion.pt": root / selected["artifact"],
                "vectorizer.joblib": root / "fusion_vectorizer.joblib",
                "scaler.joblib": root / "fusion_scaler.joblib",
            }
        )
    else:
        copies["text.joblib"] = root / selected["text_artifact"]
    for name, path in copies.items():
        shutil.copyfile(path, folder / name)
    split_manifest = Path(config["output_root"]) / "splits/manifest.json"
    manifest = {
        "schema_version": 1,
        "model_version": f"0.1.0-{selected['name']}-{file_hash(split_manifest)[:8]}",
        "kind": selected["kind"],
        "selected_model": selected["name"],
        "threshold": selected["threshold"],
        "image_size": config["image_size"],
        "text_fields": config["text_fields"],
        "strip_target_terms": config["strip_target_terms"],
        "preprocessing_version": PREPROCESS_VERSION,
        "label_version": LABEL_VERSION,
        "classes": ["proxy_negative", "proxy_abnormal"],
        "calibrated": False,
        "selection": "Highest validation study-level ROC-AUC among multimodal candidates; threshold minimizes assumed validation FN/FP cost.",
        "fn_cost": config["fn_cost"],
        "fp_cost": config["fp_cost"],
        "split_manifest_sha256": file_hash(split_manifest),
        "revision": revision(),
        "source_sha256": source_digest(),
        "files": {name: file_hash(folder / name) for name in copies},
        "comparison_artifact_sha256": {
            path.name: file_hash(path)
            for path in root.iterdir()
            if path.suffix in {".json", ".pt", ".joblib"}
        },
    }
    if selected["kind"] == "late":
        manifest["image_weight"] = selected["image_weight"]
    save_json(folder / "manifest.json", manifest)
    # Check all artifacts really load before describing this as a serving bundle.
    InferenceEngine(folder)
    return manifest


def candidate_probabilities(item, frame, config):
    root = Path(config["output_root"]) / "models"
    if item["kind"] == "text":
        return joblib.load(root / item["artifact"]).predict_proba(frame.model_text)[:, 1]
    backbone = item["name"] if item["kind"] == "image" else item["backbone"]
    encoder, _ = load_encoder(root / f"{backbone}.pt", device_for(config))
    if item["kind"] == "image":
        return predictions(encoder, frame, config)
    if item["kind"] == "late":
        text_p = joblib.load(root / item["text_artifact"]).predict_proba(frame.model_text)[:, 1]
        return (
            item["image_weight"] * predictions(encoder, frame, config)
            + (1 - item["image_weight"]) * text_p
        )
    embedding = predictions(encoder, frame, config, embeddings=True)
    tfidf = joblib.load(root / "fusion_vectorizer.joblib")
    scaler = joblib.load(root / "fusion_scaler.joblib")
    values = scaler.transform(
        np.concatenate([embedding, tfidf.transform(frame.model_text).toarray()], axis=1)
    ).astype(np.float32)
    cp = torch.load(root / item["artifact"], map_location="cpu", weights_only=True)
    fusion = FusionHead(cp["input_dim"], cp["hidden"], cp["dropout"])
    fusion.load_state_dict(cp["state_dict"])
    fusion.eval()
    with torch.inference_mode():
        return fusion(torch.from_numpy(values)).softmax(1)[:, 1].numpy()


def evaluate_test(config):
    folder = Path(config["reports_root"]) / "final"
    folder.mkdir(parents=True, exist_ok=True)
    lock = folder / "test_evaluation.lock.json"
    if lock.exists():
        raise RuntimeError(
            "Final evaluation has already been started. Preserve this evidence; use a separately documented new experiment for any rerun."
        )
    bundle = Path(config["output_root"]) / "bundle"
    manifest = json.loads((bundle / "manifest.json").read_text())
    if (
        file_hash(Path(config["output_root"]) / "splits/manifest.json")
        != manifest["split_manifest_sha256"]
    ):
        raise ValueError("Splits differ from the frozen deployment selection")
    registry = read_registries(config)
    for name, expected in manifest["comparison_artifact_sha256"].items():
        if file_hash(Path(config["output_root"]) / "models" / name) != expected:
            raise ValueError(f"Model comparison artifact changed after selection: {name}")
    with lock.open("x", encoding="utf-8") as handle:
        json.dump(
            {
                "status": "started",
                "bundle_sha256": file_hash(bundle / "manifest.json"),
                "revision": revision(),
            },
            handle,
            indent=2,
        )
    frame = load_splits(config, include_test=True)["test"]
    rows = []
    for name, item in registry.items():
        with tracked(config, "final-evaluation", name):
            probs = candidate_probabilities(item, frame, config)
            result = evidence(frame, probs, item["threshold"], folder / name, "test")
            rows.append(
                {
                    "model": name,
                    "family": item["family"],
                    **{k: v for k, v in result.items() if k != "confusion_matrix"},
                }
            )
            mlflow.log_metrics(
                {f"test_{k}": v for k, v in result.items() if isinstance(v, (int, float))}
            )
            mlflow.log_artifacts(str(folder / name))
    table = pd.DataFrame(rows)
    table.to_csv(folder / "ablation_results.csv", index=False)
    axis = table.set_index("model")[["roc_auc", "f1"]].plot.bar(figsize=(11, 5), ylim=(0, 1))
    axis.figure.tight_layout()
    axis.figure.savefig(folder / "ablation.png", dpi=140)
    import matplotlib.pyplot as plt

    plt.close(axis.figure)
    save_json(
        lock,
        {
            "status": "complete",
            "bundle_sha256": file_hash(bundle / "manifest.json"),
            "selected_before_test": manifest["selected_model"],
            "revision": revision(),
        },
    )
    return rows


def explanation_examples(config, limit=20):
    from pulmonary.explainability import explain

    engine = InferenceEngine(Path(config["output_root"]) / "bundle")
    frame = load_splits(config)["validation"]
    folder = Path(config["reports_root"]) / "xai_examples"
    folder.mkdir(parents=True, exist_ok=True)
    import base64

    groups = {name: [] for name in ["TN", "TP", "FP", "FN"]}
    eligible = frame.drop_duplicates("uid")
    eligible = eligible.loc[eligible.model_text.str.strip().ne("")].sample(
        frac=1, random_state=config["seed"]
    )
    for _, row in eligible.iterrows():
        with Image.open(Path(config["data_root"]) / row.image_path) as image:
            prediction = engine.predict(
                image, str(row.get("indication", "")), str(row.get("comparison", ""))
            )
        positive = prediction["predicted_label"] == "proxy_abnormal"
        category = ("TP" if positive else "FN") if row.target == 1 else ("FP" if positive else "TN")
        groups[category].append(row)
    desired = {name: limit // 4 + int(index < limit % 4) for index, name in enumerate(groups)}
    selected = []
    for name, cases in groups.items():
        selected.extend((name, row) for row in cases[: desired[name]])
    used = {row.sample_id for _, row in selected}
    for name, cases in groups.items():
        for row in cases:
            if len(selected) >= limit:
                break
            if row.sample_id not in used:
                selected.append((name, row))
                used.add(row.sample_id)
    for count, (category, row) in enumerate(selected):
        with Image.open(Path(config["data_root"]) / row.image_path) as image:
            result = explain(
                engine, image, str(row.get("indication", "")), str(row.get("comparison", ""))
            )
        (folder / f"case_{count:02d}.png").write_bytes(
            base64.b64decode(result.pop("gradcam_base64"))
        )
        result.update(
            {
                "uid": str(row.uid),
                "weak_target": int(row.target),
                "source": "validation",
                "selection_category": category,
            }
        )
        save_json(folder / f"case_{count:02d}.json", result)
        pred = result["prediction"]
        lines = [
            f"# Validation study {row.uid} — {category}",
            "",
            f"Weak target: {int(row.target)}; model score: {pred['probability']:.4f}; threshold: {pred['threshold']:.4f}; priority: {pred['priority_tier']}.",
            "",
            f"![Final multimodal score Grad-CAM](case_{count:02d}.png)",
            "",
            "```json",
            json.dumps(
                {"indication": row.get("indication", ""), "comparison": row.get("comparison", "")},
                ensure_ascii=False,
            ),
            "```",
            "",
            "| Token | Full score minus score after removal |",
            "| --- | --- |",
        ]
        lines.extend(
            f"| {item['token']} | {item['contribution']:+.5f} |"
            for item in result["text_highlights"][:10]
        )
        lines += [
            "",
            "Positive contribution means the token's presence raises the score. These context-dependent changes are not additive causal effects. The weak target is not clinical ground truth.",
        ]
        (folder / f"case_{count:02d}.md").write_text("\n".join(lines), encoding="utf-8")
    selection = {
        "examples_saved": len(selected),
        "split": "validation",
        "available_single_image_cases": {name: len(cases) for name, cases in groups.items()},
        "desired_category_counts": desired,
        "saved_category_counts": {
            name: sum(category == name for category, _ in selected) for name in groups
        },
        "selection": "Seeded order; aim for balanced TN/TP/FP/FN against weak labels; fill unavailable quotas with other cases. Uses one image per study, matching the single-image API.",
    }
    save_json(folder / "selection_manifest.json", selection)
    return selection
