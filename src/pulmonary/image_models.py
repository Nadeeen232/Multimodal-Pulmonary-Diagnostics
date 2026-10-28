"""Two-stage image transfer learning and stable ordered embedding extraction."""

import copy
import logging
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from pulmonary.common import (
    device_for,
    load_splits,
    require_verified_images,
    save_json,
    seed_everything,
    split_digest,
)
from pulmonary.evaluation import choose_threshold, cost_evidence, evidence
from pulmonary.images import image_tensor
from pulmonary.networks import ImageEncoder, load_encoder
from pulmonary.precision import grad_scaler, use_amp, validate_training_limits
from pulmonary.tracking import log_results, tracked


class RadiographDataset(Dataset):
    def __init__(self, frame, config, augment=False):
        self.frame = frame.reset_index(drop=True)
        self.root = Path(config["data_root"])
        self.output_root = Path(config["output_root"])
        self.size = config["image_size"]
        self.augment = augment

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        path = (
            self.output_root / row.processed_image_path
            if row.get("processed_image_path", "")
            else self.root / row.image_path
        )
        with Image.open(path) as image:
            tensor = image_tensor(image, self.size, self.augment)
        return tensor, torch.tensor(int(row.target), dtype=torch.long)


def loader_for(frame, config, train=False):
    return DataLoader(
        RadiographDataset(frame, config, train),
        batch_size=min(config["batch_size"], 8),
        shuffle=train,
        num_workers=config["num_workers"],
        generator=torch.Generator().manual_seed(config["seed"]),
    )


def predictions(model, frame, config, embeddings=False):
    device = next(model.parameters()).device
    values = []
    model.eval()
    with torch.inference_mode():
        for images, _ in loader_for(frame, config):
            images = images.to(device)
            result = model.embed(images) if embeddings else model(images).softmax(1)[:, 1]
            values.append(result.cpu().numpy())
    return np.concatenate(values)


def configure_stage(model, phase):
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    if phase == "finetune":
        blocks = (
            list(model.features.children())[-2:]
            if model.name == "resnet50"
            else list(model.features[0].children())[-5:]
        )
        for block in blocks:
            for parameter in block.parameters():
                parameter.requires_grad = True
    elif phase != "head":
        raise ValueError(f"Unknown training phase: {phase}")


def train_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    scaler=None,
    amp_enabled=False,
    amp_statistics=None,
):
    training = optimizer is not None
    model.train(training)
    if training:
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
    losses, weight_mass = 0.0, 0.0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, targets)
            if amp_enabled and not torch.isfinite(loss):
                logging.getLogger(__name__).warning(
                    "Nonfinite AMP forward pass; recomputing this batch in FP32"
                )
                if amp_statistics is not None:
                    amp_statistics["fp32_fallback_batches"] = (
                        amp_statistics.get("fp32_fallback_batches", 0) + 1
                    )
                del logits, loss
                with torch.autocast(device_type=device.type, enabled=False):
                    logits = model(images.float())
                    loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Nonfinite FP32 training/validation loss; stopping before saving an invalid model"
                )
            if training:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()
                nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
        mass = (
            float(criterion.weight[targets].sum()) if criterion.weight is not None else len(targets)
        )
        losses += float(loss.detach()) * mass
        weight_mass += mass
    return losses / weight_mass


def train_images(config):
    validate_training_limits(config)
    require_verified_images(config)
    seed_everything(config["seed"])
    splits = load_splits(config)
    train, val = splits["train"], splits["validation"]
    device = device_for(config)
    amp_enabled = use_amp(config, device)
    model_root = Path(config["output_root"]) / "models"
    model_root.mkdir(parents=True, exist_ok=True)
    counts = np.bincount(train.target, minlength=2)
    weights = torch.tensor(len(train) / (2 * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    registry = {}
    for name in config["image_models"]:
        seed_everything(config["seed"])
        train_loader, val_loader = loader_for(train, config, True), loader_for(val, config)
        with tracked(
            config,
            "cnn",
            name,
            {
                "backbone": name,
                "pretrained": True,
                "batch_size": config["batch_size"],
                "dropout": config["dropout"],
                "amp": amp_enabled,
                "max_epochs": config.get("max_epochs", 20),
            },
        ):
            # Download errors fail visibly. Frozen random features are not a transfer-learning fallback.
            model = ImageEncoder(name, pretrained=True, dropout=config["dropout"]).to(device)
            scaler = grad_scaler(amp_enabled)
            amp_statistics = {"fp32_fallback_batches": 0}
            best_loss, best_state, history, step = float("inf"), None, [], 0
            for phase, epochs in [
                ("head", config["head_epochs"]),
                ("finetune", config["finetune_epochs"]),
            ]:
                configure_stage(model, phase)
                optimizer = torch.optim.AdamW(
                    [
                        {
                            "params": [p for p in model.features.parameters() if p.requires_grad],
                            "lr": config["learning_rate"] / 10,
                        },
                        {"params": list(model.head.parameters()), "lr": config["learning_rate"]},
                    ],
                    weight_decay=config["weight_decay"],
                )
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, patience=1, factor=0.3
                )
                wait = 0
                for _ in range(epochs):
                    train_loss = train_epoch(
                        model,
                        train_loader,
                        criterion,
                        device,
                        optimizer,
                        scaler=scaler,
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
                    scheduler.step(val_loss)
                    row = {
                        "epoch": step,
                        "phase": phase,
                        "train_loss": train_loss,
                        "validation_loss": val_loss,
                    }
                    history.append(row)
                    mlflow.log_metrics(
                        {"train_loss": train_loss, "validation_loss": val_loss, **amp_statistics},
                        step=step,
                    )
                    print(
                        f"{name} {phase} epoch={step + 1} train_loss={train_loss:.4f} val_loss={val_loss:.4f}",
                        flush=True,
                    )
                    step += 1
                    if val_loss < best_loss - 1e-4:
                        best_loss = val_loss
                        best_state = copy.deepcopy(
                            {k: v.cpu() for k, v in model.state_dict().items()}
                        )
                        wait = 0
                    else:
                        wait += 1
                    if phase == "finetune" and wait >= config["patience"]:
                        break
            if best_state is None:
                raise ValueError("No image training epochs completed")
            del optimizer, scheduler
            model.load_state_dict(best_state)
            probs = predictions(model, val, config)
            threshold, table = choose_threshold(val, probs, config)
            folder = Path(config["reports_root"]) / name
            result = evidence(val, probs, threshold, folder, "validation")
            pd.DataFrame(history).to_csv(folder / "learning_history.csv", index=False)
            cost_evidence(table, folder)
            path = model_root / f"{name}.pt"
            torch.save(
                {
                    "state_dict": best_state,
                    "backbone": name,
                    "dropout": config["dropout"],
                    "image_size": config["image_size"],
                    "amp_training": amp_enabled,
                },
                path,
            )
            log_results(result, folder, path)
            registry[name] = {
                "name": name,
                "family": "cnn",
                "kind": "image",
                "artifact": path.name,
                "threshold": threshold,
                "validation": result,
            }
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    best = max(registry, key=lambda key: registry[key]["validation"]["roc_auc"])
    save_json(
        model_root / "image_registry.json",
        {"best": best, "models": registry, "split_manifest_sha256": split_digest(config)},
    )
    return registry


def extract_embeddings(config, backbone, splits):
    model_root = Path(config["output_root"]) / "models"
    model, _ = load_encoder(model_root / f"{backbone}.pt", device_for(config))
    folder = Path(config["output_root"]) / "embeddings"
    folder.mkdir(parents=True, exist_ok=True)
    result = {}
    for name, frame in splits.items():
        values = predictions(model, frame, config, embeddings=True)
        np.savez_compressed(
            folder / f"{backbone}_{name}.npz",
            embedding=values,
            sample_id=frame.sample_id.to_numpy(dtype=str),
        )
        result[name] = values
    return model, result
