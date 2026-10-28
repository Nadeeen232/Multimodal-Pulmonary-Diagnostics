"""File-based reproduction of the team's selected late-fusion approach.

Uses an isolated cohort and artifacts; never overwrites the deployed v1 bundle.
Reported notebook metrics remain historical until this pipeline is evaluated.
"""

import gc
import json
import shutil
import time
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from pulmonary.common import file_hash, save_json, seed_everything, source_digest
from pulmonary.data import link_duplicate_groups
from pulmonary.image_models import train_epoch
from pulmonary.precision import grad_scaler
from pulmonary.team_features import (
    PREPROCESS,
    SIZE,
    TeamTextModel,
    build_model,
    image_transforms,
    letterbox,
)
from pulmonary.team_rules import (
    CHRONIC_CORE,
    CORE_FLAGS,
    NO_ACUTE_PATTERN,
    clean_section,
    derive_labels,
    iterative_multilabel_split,
)


def prepare(data_root, output):
    """Reproduce labels, audit pairing, and prevent exact-image split leakage."""
    root, output = Path(data_root).resolve(), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    reports_path, projections_path = [], []
    for name, matches in [
        ("indiana_reports.csv", reports_path),
        ("indiana_projections.csv", projections_path),
    ]:
        matches.extend(root.rglob(name))
        if len(matches) != 1:
            raise ValueError(f"Expected one {name}, found {len(matches)}")
    reports = pd.read_csv(reports_path[0], dtype={"uid": str})
    projections = pd.read_csv(projections_path[0], dtype={"uid": str})
    reports = reports.drop_duplicates()
    if reports.uid.isna().any() or reports.uid.duplicated().any():
        raise ValueError("Conflicting or missing report UID")
    if projections.filename.duplicated().any():
        raise ValueError("Image filenames must map to one study/view")
    index = {}
    for path in root.rglob("*"):
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            if path.name in index:
                raise ValueError(f"Ambiguous image basename: {path.name}")
            index[path.name] = path
    projection = projections.projection.fillna("").str.strip().str.lower()
    for column, views in [("frontal_img", {"frontal", "ap", "pa"}), ("lateral_img", {"lateral"})]:
        mapping = (
            projections.loc[projection.isin(views)].drop_duplicates("uid").set_index("uid").filename
        )
        reports[column] = reports.uid.map(mapping)
    reports["impression_clean"] = reports.impression.map(lambda v: clean_section(v)["clean"])
    frame = derive_labels(reports.loc[reports.frontal_img.isin(index)].copy())
    core = [f"lbl_{g}" for g in CORE_FLAGS]
    nonchronic = [f"lbl_{g}" for g in CORE_FLAGS if g not in CHRONIC_CORE]
    abnormal = frame[core].sum(axis=1).gt(0)
    first_sentence = frame.impression_clean.str.split(".").str[0].str.lower()
    downgrade = (
        abnormal
        & frame[nonchronic].sum(axis=1).eq(0)
        & first_sentence.str.contains(NO_ACUTE_PATTERN, regex=True)
        & frame.is_normal_tag.eq(0)
    )
    frame["task_label"] = "uncertain"
    frame.loc[frame.is_normal_tag.eq(1) & ~abnormal, "task_label"] = "normal"
    frame.loc[abnormal & frame.is_normal_tag.eq(0), "task_label"] = "abnormal"
    frame.loc[downgrade, "task_label"] = "uncertain"
    frame.drop(columns=["findings", "impression", "impression_clean"]).to_csv(
        output / "cohort_audit.csv", index=False
    )
    binary = (
        frame.loc[frame.task_label.ne("uncertain") & frame.poor_quality.eq(0)]
        .copy()
        .reset_index(drop=True)
    )
    binary["target"] = binary.task_label.eq("abnormal").astype(int)
    split_cols = ["target"] + [c for c in core if binary[c].sum() > 0]
    binary["split"] = iterative_multilabel_split(binary[split_cols].to_numpy(), seed=42)
    rows = []
    cache = output / "images_384"
    cache.mkdir(exist_ok=True)
    for number, row in enumerate(binary.to_dict("records"), 1):
        for field, view in [("frontal_img", "Frontal"), ("lateral_img", "Lateral")]:
            name = row[field]
            if pd.isna(name):
                continue
            if name not in index:
                raise FileNotFoundError(f"Missing paired image: {name}")
            digest = file_hash(index[name])
            destination = cache / f"{digest}.png"
            with Image.open(index[name]) as image:
                image.load()
                if not destination.exists():
                    letterbox(image).save(destination)
            rows.append(
                {
                    "uid": row["uid"],
                    "group_id": row["uid"],
                    "target": row["target"],
                    "split": row["split"],
                    "filename": name,
                    "projection": view,
                    "image_sha256": digest,
                    "processed_image_path": destination.relative_to(output).as_posix(),
                }
            )
        if number % 250 == 0:
            print(f"Prepared {number}/{len(binary)} studies", flush=True)
    images, duplicate_rows = link_duplicate_groups(pd.DataFrame(rows))
    crossing = images.groupby("group_id").split.nunique().gt(1)
    if crossing.any():
        uid_group = images.drop_duplicates("uid").set_index("uid").group_id
        binary["group_id"] = binary.uid.map(uid_group)
        groups = binary.groupby("group_id", sort=False)[split_cols].max()
        assignment = pd.Series(
            iterative_multilabel_split(groups.to_numpy(), seed=42), index=groups.index
        )
        binary["split"] = binary.group_id.map(assignment)
        images["split"] = images.group_id.map(assignment)
    binary = binary.drop(columns=["findings", "impression", "impression_clean"], errors="ignore")
    binary.to_csv(output / "studies.csv", index=False)
    images.to_csv(output / "images.csv", index=False)
    manifest = {
        "preprocessing": PREPROCESS,
        "seed": 42,
        "strict_downgrades": int(downgrade.sum()),
        "duplicate_image_rows": duplicate_rows,
        "cross_split_duplicate_components": int(crossing.sum()),
        "split_changed_for_duplicates": bool(crossing.any()),
        "patient_independence_verified": False,
        "previous_project_test_already_seen": True,
        "study_counts": binary.split.value_counts().to_dict(),
        "image_counts": images.split.value_counts().to_dict(),
        "csv_sha256": {
            "reports": file_hash(reports_path[0]),
            "projections": file_hash(projections_path[0]),
        },
        "manifest_sha256": {
            "studies": file_hash(output / "studies.csv"),
            "images": file_hash(output / "images.csv"),
        },
    }
    save_json(output / "cohort_manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def threshold(y, probability):
    grid = np.arange(0.05, 0.951, 0.01)
    return float(grid[np.argmax([balanced_accuracy_score(y, probability >= t) for t in grid])])


def metrics(y, probability, cutoff):
    return {
        "accuracy": accuracy_score(y, probability >= cutoff),
        "f1": f1_score(y, probability >= cutoff),
        "roc_auc": roc_auc_score(y, probability),
        "threshold": cutoff,
        "studies": len(y),
    }


def train_text(output):
    output = Path(output)
    frame = pd.read_csv(output / "studies.csv", dtype={"uid": str}, keep_default_na=False)
    train, val = (frame.loc[frame.split.eq(s)] for s in ["train", "val"])
    model = TeamTextModel()
    x_train, x_val = model.matrix(train, fit=True), model.matrix(val)
    candidates = []
    best_auc = -1
    for c in [0.03, 0.1, 0.3, 1, 3, 10]:
        for weight in [None, "balanced"]:
            classifier = LogisticRegression(
                C=c, class_weight=weight, max_iter=2000, solver="liblinear", random_state=42
            ).fit(x_train, train.target)
            auc = roc_auc_score(val.target, classifier.predict_proba(x_val)[:, 1])
            candidates.append({"C": c, "class_weight": weight, "validation_auc": auc})
            if auc > best_auc:
                best_auc, model.classifier = auc, classifier
    joblib.dump(model, output / "text_model.joblib")
    pd.DataFrame(candidates).to_csv(output / "text_search.csv", index=False)
    save_json(
        output / "text_selection.json",
        {
            "validation_auc": best_auc,
            "params": model.classifier.get_params(),
            "training_manifest": file_hash(output / "studies.csv"),
        },
    )
    print(f"TF-IDF validation AUC: {best_auc:.4f}", flush=True)


class TeamImages(Dataset):
    def __init__(self, frame, root, augment=False):
        self.frame, self.root, self.transform = (
            frame.reset_index(drop=True),
            Path(root),
            image_transforms(augment),
        )

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(self.root / row.processed_image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, torch.tensor(int(row.target), dtype=torch.long)


def loader(frame, root, augment=False):
    return DataLoader(
        TeamImages(frame, root, augment),
        batch_size=8,
        shuffle=augment,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(42),
    )


@torch.inference_mode()
def predict_images(net, frame, output, device):
    net.eval()
    values = []
    for images, _ in loader(frame, output):
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = net(images.to(device))
        values.extend(logits.float().softmax(1)[:, 1].cpu().numpy())
    return np.asarray(values)


def train_vision(output, name):
    output = Path(output)
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # Tested at batch 8 on this GTX 1660 SUPER. Deterministic cuDNN FP16
        # returned NaNs; legacy cuDNN with benchmark selection has finite
        # forward/backward passes and avoids the much slower native kernels.
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    frame = pd.read_csv(output / "images.csv", dtype={"uid": str})
    train, val = (frame.loc[frame.split.eq(s)] for s in ["train", "val"])
    net = build_model(name, pretrained=True).to(device)
    checkpoint_path = output / f"{name}.pt"
    previous = None
    if checkpoint_path.exists():
        previous = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if previous["cohort_manifest"] != file_hash(output / "cohort_manifest.json"):
            raise ValueError("Cannot resume a checkpoint from a different cohort")
        net.load_state_dict(previous["state_dict"])
        if (output / f"{name}_development_predictions.csv").exists():
            print(
                f"{name}: completed checkpoint and development predictions already exist",
                flush=True,
            )
            return
    head = net.fc if name == "resnet50" else net.classifier
    blocks = (
        [net.layer3, net.layer4]
        if name == "resnet50"
        else [
            net.features.denseblock3,
            net.features.transition3,
            net.features.denseblock4,
            net.features.norm5,
        ]
    )
    weights = torch.tensor(
        len(train) / (2 * np.bincount(train.target, minlength=2)),
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=weights)
    scaler = grad_scaler(device.type == "cuda")
    train_loader, val_loader = loader(train, output, True), loader(val, output)
    history, best_loss, best_state, wait = [], float("inf"), None, 0
    if previous is not None:
        history = pd.read_csv(output / f"{name}_history.csv").to_dict("records")
        best_state = previous["state_dict"]
        best_loss = float(
            next(row["val_loss"] for row in history if row["epoch"] == previous["epoch"])
        )
        print(
            f"Resuming {name} from best epoch {previous['epoch']}; optimizer reinitialized",
            flush=True,
        )
    mlflow.set_tracking_uri(f"sqlite:///{(output / 'mlflow.db').resolve().as_posix()}")
    mlflow.set_experiment("team29-classification-reproduction")
    with mlflow.start_run(run_name=name):
        mlflow.log_params(
            {
                "model": name,
                "batch_size": 8,
                "amp": device.type == "cuda",
                "image_size": SIZE,
                "seed": 42,
                "preprocess": PREPROCESS,
                "source_digest": source_digest(),
                "cohort_manifest": file_hash(output / "cohort_manifest.json"),
                "max_epochs": 18,
                "cudnn_enabled": torch.backends.cudnn.enabled,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "resumed_checkpoint_epoch": previous["epoch"] if previous else 0,
            }
        )
        for phase, epochs in [("head", 3), ("finetune", 15)]:
            epochs = max(0, 3 - len(history)) if phase == "head" else max(0, 18 - len(history))
            if epochs == 0:
                continue
            for parameter in net.parameters():
                parameter.requires_grad = False
            for parameter in head.parameters():
                parameter.requires_grad = True
            groups = [{"params": list(head.parameters()), "lr": 3e-4 if phase == "head" else 1e-4}]
            if phase == "finetune":
                backbone_parameters = [p for block in blocks for p in block.parameters()]
                for parameter in backbone_parameters:
                    parameter.requires_grad = True
                groups.append({"params": backbone_parameters, "lr": 1e-5})
                wait = 0
            optimizer = torch.optim.AdamW(groups, weight_decay=2e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=0.3, patience=1, min_lr=1e-7
            )
            for _ in range(epochs):
                if shutil.disk_usage(output).free < 256 * 1024**2:
                    raise RuntimeError(
                        "Training stopped before the next epoch: free at least 512 MB of disk space, then resume"
                    )
                start = time.perf_counter()
                tr = train_epoch(
                    net, train_loader, criterion, device, optimizer, scaler, device.type == "cuda"
                )
                va = train_epoch(
                    net, val_loader, criterion, device, amp_enabled=device.type == "cuda"
                )
                scheduler.step(va)
                epoch = len(history) + 1
                entry = {
                    "epoch": epoch,
                    "phase": phase,
                    "train_loss": tr,
                    "val_loss": va,
                    "seconds": time.perf_counter() - start,
                }
                history.append(entry)
                save_json(output / "progress.json", {"model": name, **entry})
                mlflow.log_metrics({"train_loss": tr, "validation_loss": va}, step=epoch)
                print(
                    f"{name} {phase} epoch {epoch}/18 train_loss={tr:.5f} val_loss={va:.5f} seconds={entry['seconds']:.1f}",
                    flush=True,
                )
                if va < best_loss - 1e-4:
                    best_loss, wait = va, 0
                    best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                    temporary = checkpoint_path.with_suffix(".pt.partial")
                    torch.save(
                        {
                            "state_dict": best_state,
                            "model": name,
                            "preprocess": PREPROCESS,
                            "epoch": epoch,
                            "validation_loss": va,
                            "cohort_manifest": file_hash(output / "cohort_manifest.json"),
                        },
                        temporary,
                    )
                    temporary.replace(checkpoint_path)
                else:
                    wait += 1
                pd.DataFrame(history).to_csv(output / f"{name}_history.csv", index=False)
                if phase == "finetune" and wait >= 2:
                    break
        net.load_state_dict(best_state)
        # Test data is deliberately absent from the training/prediction stage.
        predictions = []
        for split in ["train", "val"]:
            part = frame.loc[frame.split.eq(split)].copy()
            part["probability"] = predict_images(net, part, output, device)
            predictions.append(part)
        pd.concat(predictions).to_csv(output / f"{name}_development_predictions.csv", index=False)
        mlflow.log_artifact(str(output / f"{name}_history.csv"))
    del net
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def study_scores(frame):
    if frame.groupby("uid").target.nunique().gt(1).any():
        raise ValueError("Conflicting study labels")
    return frame.groupby("uid", sort=True).agg(
        target=("target", "first"), probability=("probability", "mean")
    )


def finalize(output):
    """Freeze all choices on validation, then evaluate once on the isolated test."""
    output = Path(output)
    if (output / "test_evaluation_lock.json").exists():
        raise RuntimeError(
            "Test evaluation already exists; inspect saved results instead of retuning"
        )
    studies = pd.read_csv(output / "studies.csv", dtype={"uid": str}, keep_default_na=False)
    text = joblib.load(output / "text_model.joblib")
    val = studies.loc[studies.split.eq("val")].set_index("uid").sort_index()
    p_text = text.predict_proba(val)[:, 1]
    scores, cutoffs = {}, {}
    for name in ["resnet50", "densenet121"]:
        dev = pd.read_csv(output / f"{name}_development_predictions.csv", dtype={"uid": str})
        scores[name] = (
            study_scores(dev.loc[dev.split.eq("val")]).reindex(val.index).probability.to_numpy()
        )
        if not np.isfinite(scores[name]).all():
            raise ValueError("Incomplete development image/text alignment")
    scores["image_ensemble"] = (scores["resnet50"] + scores["densenet121"]) / 2
    scores["text_only"] = p_text
    candidates = [
        (roc_auc_score(val.target, a * scores["image_ensemble"] + (1 - a) * p_text), a)
        for a in np.linspace(0, 1, 21)
    ]
    _, alpha = max(candidates, key=lambda value: (value[0], -value[1]))
    scores["multimodal_fusion"] = alpha * scores["image_ensemble"] + (1 - alpha) * p_text
    for name, values in scores.items():
        cutoffs[name] = threshold(val.target, values)
    selection = {
        "image_weight": float(alpha),
        "thresholds": cutoffs,
        "objective": "validation ROC-AUC; validation balanced-accuracy thresholds",
        "cohort_manifest": file_hash(output / "cohort_manifest.json"),
        "status": "choices frozen before test evaluation",
    }
    save_json(output / "selection.json", selection)
    save_json(output / "test_evaluation_lock.json", selection)
    images = pd.read_csv(output / "images.csv", dtype={"uid": str})
    test_images = images.loc[images.split.eq("test")]
    test = studies.loc[studies.split.eq("test")].set_index("uid").sort_index()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_scores = {"text_only": text.predict_proba(test)[:, 1]}
    for name in ["resnet50", "densenet121"]:
        checkpoint = torch.load(output / f"{name}.pt", weights_only=True, map_location="cpu")
        if checkpoint["cohort_manifest"] != selection["cohort_manifest"]:
            raise ValueError("Checkpoint trained on a different cohort")
        net = build_model(name).to(device)
        net.load_state_dict(checkpoint["state_dict"])
        predictions = test_images.copy()
        predictions["probability"] = predict_images(net, predictions, output, device)
        test_scores[name] = study_scores(predictions).reindex(test.index).probability.to_numpy()
        predictions.to_csv(output / f"{name}_test_predictions.csv", index=False)
        del net
    test_scores["image_ensemble"] = (test_scores["resnet50"] + test_scores["densenet121"]) / 2
    test_scores["multimodal_fusion"] = (
        alpha * test_scores["image_ensemble"] + (1 - alpha) * test_scores["text_only"]
    )
    results = [
        {"model": name, **metrics(test.target, values, cutoffs[name])}
        for name, values in test_scores.items()
    ]
    pd.DataFrame(results).to_csv(output / "ablation_results.csv", index=False)
    for name, values in test_scores.items():
        test[f"p_{name}"] = values
    test.to_csv(output / "study_test_predictions.csv")
    print(pd.DataFrame(results).to_string(index=False), flush=True)
