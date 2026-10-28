"""Frozen image embeddings + text MLP and bounded late probability fusion."""

import copy
import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pulmonary.common import (
    device_for,
    load_splits,
    require_verified_images,
    save_json,
    seed_everything,
    split_digest,
)
from pulmonary.evaluation import choose_threshold, cost_evidence, evidence, metrics
from pulmonary.image_models import extract_embeddings, predictions, train_epoch
from pulmonary.networks import FusionHead
from pulmonary.precision import grad_scaler, use_amp, validate_training_limits
from pulmonary.tracking import log_results, tracked


def train_fusion(config):
    validate_training_limits(config)
    require_verified_images(config)
    seed_everything(config["seed"])
    splits = load_splits(config)
    train, val = splits["train"], splits["validation"]
    root = Path(config["output_root"]) / "models"
    images = json.loads((root / "image_registry.json").read_text())
    texts = json.loads((root / "text_registry.json").read_text())
    if any(item["split_manifest_sha256"] != split_digest(config) for item in [images, texts]):
        raise ValueError("Baseline models were trained on different splits; retrain before fusion")
    backbone = images["best"]
    image_model, embeddings = extract_embeddings(config, backbone, splits)
    image_model.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    vectorizer = TfidfVectorizer(
        max_features=config["max_features"], ngram_range=(1, 2), sublinear_tf=True
    )
    vectorizer.fit(train.drop_duplicates("uid").model_text)
    scaler = StandardScaler()
    features = {}
    for name, frame in splits.items():
        values = np.concatenate(
            [embeddings[name], vectorizer.transform(frame.model_text).toarray()], axis=1
        )
        features[name] = (
            scaler.fit_transform(values) if name == "train" else scaler.transform(values)
        ).astype(np.float32)
    joblib.dump(vectorizer, root / "fusion_vectorizer.joblib")
    joblib.dump(scaler, root / "fusion_scaler.joblib")
    device = device_for(config)
    amp_enabled = use_amp(config, device)
    registry = {}
    with tracked(
        config,
        "multimodal",
        "fusion_early",
        {
            "backbone": backbone,
            "hidden": config["fusion_hidden"],
            "dropout": config["dropout"],
            "amp": amp_enabled,
            "batch_size": 8,
            "max_epochs": config.get("max_epochs", 20),
        },
    ):
        model = FusionHead(
            features["train"].shape[1], config["fusion_hidden"], config["dropout"]
        ).to(device)
        counts = np.bincount(train.target, minlength=2)
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(len(train) / (2 * counts), dtype=torch.float32, device=device)
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
        )
        loader = DataLoader(
            TensorDataset(
                torch.from_numpy(features["train"]),
                torch.tensor(train.target.to_numpy(), dtype=torch.long),
            ),
            batch_size=config["batch_size"],
            shuffle=True,
            generator=torch.Generator().manual_seed(config["seed"]),
        )
        val_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(features["validation"]),
                torch.tensor(val.target.to_numpy(), dtype=torch.long),
            ),
            batch_size=8,
            shuffle=False,
        )
        scaler_amp = grad_scaler(amp_enabled)
        amp_statistics = {"fp32_fallback_batches": 0}
        best, best_state, wait = float("inf"), None, 0
        for epoch in range(config["fusion_epochs"]):
            train_loss = train_epoch(
                model,
                loader,
                criterion,
                device,
                optimizer,
                scaler=scaler_amp,
                amp_enabled=amp_enabled,
                amp_statistics=amp_statistics,
            )
            val_loss = train_epoch(
                model,
                val_loader,
                criterion,
                device,
                amp_enabled=amp_enabled,
                amp_statistics=amp_statistics,
            )
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metrics(amp_statistics, step=epoch)
            mlflow.log_metric("validation_loss", val_loss, step=epoch)
            print(f"fusion epoch={epoch + 1} val_loss={val_loss:.4f}", flush=True)
            if val_loss < best - 1e-4:
                best, best_state, wait = (
                    val_loss,
                    copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()}),
                    0,
                )
            else:
                wait += 1
            if wait >= config["patience"]:
                break
        if best_state is None:
            raise ValueError("No fusion training epochs completed")
        model.load_state_dict(best_state)
        model.eval()
        with torch.inference_mode():
            probs = np.concatenate(
                [model(x.to(device)).softmax(1)[:, 1].cpu().numpy() for x, _ in val_loader]
            )
        threshold, costs = choose_threshold(val, probs, config)
        folder = Path(config["reports_root"]) / "fusion_early"
        result = evidence(val, probs, threshold, folder, "validation")
        cost_evidence(costs, folder)
        path = root / "fusion_early.pt"
        torch.save(
            {
                "state_dict": best_state,
                "input_dim": features["train"].shape[1],
                "hidden": config["fusion_hidden"],
                "dropout": config["dropout"],
            },
            path,
        )
        log_results(result, folder, path)
        mlflow.log_artifact(str(root / "fusion_vectorizer.joblib"), artifact_path="model")
        mlflow.log_artifact(str(root / "fusion_scaler.joblib"), artifact_path="model")
        registry["fusion_early"] = {
            "name": "fusion_early",
            "family": "multimodal",
            "kind": "early",
            "backbone": backbone,
            "artifact": path.name,
            "threshold": threshold,
            "validation": result,
        }
    with tracked(
        config, "multimodal", "fusion_late", {"backbone": backbone, "text_model": texts["best"]}
    ):
        text_model = joblib.load(root / texts["models"][texts["best"]]["artifact"])
        text_probs = text_model.predict_proba(val.model_text)[:, 1]
        image_model.to(device)
        image_probs = predictions(image_model, val, config)
        rows = []
        # Both modalities always contribute. Grid and selection criterion are fixed in advance.
        for alpha in np.linspace(0.1, 0.9, 17):
            probs = alpha * image_probs + (1 - alpha) * text_probs
            rows.append(
                {"image_weight": float(alpha), "roc_auc": metrics(val, probs, 0.5)["roc_auc"]}
            )
        chosen = max(rows, key=lambda row: row["roc_auc"])
        alpha = chosen["image_weight"]
        probs = alpha * image_probs + (1 - alpha) * text_probs
        threshold, costs = choose_threshold(val, probs, config)
        folder = Path(config["reports_root"]) / "fusion_late"
        result = evidence(val, probs, threshold, folder, "validation")
        cost_evidence(costs, folder)
        save_json(folder / "weight_search.json", rows)
        path = root / "fusion_late.json"
        save_json(path, {"image_weight": alpha, "text_model": texts["best"], "backbone": backbone})
        log_results(result, folder, path)
        registry["fusion_late"] = {
            "name": "fusion_late",
            "family": "multimodal",
            "kind": "late",
            "backbone": backbone,
            "text_artifact": texts["models"][texts["best"]]["artifact"],
            "image_weight": alpha,
            "artifact": path.name,
            "threshold": threshold,
            "validation": result,
        }
    best_name = max(registry, key=lambda key: registry[key]["validation"]["roc_auc"])
    save_json(
        root / "fusion_registry.json",
        {"best": best_name, "models": registry, "split_manifest_sha256": split_digest(config)},
    )
    return registry
