"""Serving adapter for the locally reproduced Team29 classification bundle."""

import json
import math
import threading
from pathlib import Path

import joblib
import pandas as pd
import torch
from torch import nn

from pulmonary.common import file_hash
from pulmonary.priority import get_priority
from pulmonary.team_features import PREPROCESS, build_model, image_transforms, letterbox
from pulmonary.team_rules import make_report_text


class TextInput(str):
    """Keep raw fields for metadata features while exposing cleaned text validation."""

    def __new__(cls, indication, comparison):
        obj = super().__new__(
            cls, make_report_text({"indication": indication, "comparison": comparison})
        )
        obj.indication, obj.comparison = indication, comparison
        return obj


class ImageEnsemble(nn.Module):
    def __init__(self, resnet, densenet):
        super().__init__()
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.pool = resnet.avgpool
        self.head = resnet.fc
        self.densenet = densenet

    def forward(self, tensor):
        resnet = self.head(self.pool(self.features(tensor)).flatten(1)).softmax(1)[:, 1]
        dense = self.densenet(tensor).softmax(1)[:, 1]
        return (resnet + dense) / 2


class TeamInferenceEngine:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.manifest = json.loads((self.folder / "manifest.json").read_text())
        if (
            self.manifest.get("kind") != "team29_late"
            or self.manifest.get("preprocessing_version") != PREPROCESS
        ):
            raise ValueError("Unsupported team bundle")
        if self.manifest.get("image_size") != 384:
            raise ValueError("Team model requires 384-pixel preprocessing")
        required = {"resnet50.pt", "densenet121.pt", "text_model.joblib"}
        if set(self.manifest["files"]) != required:
            raise ValueError("Incomplete team bundle")
        for name, expected in self.manifest["files"].items():
            if file_hash(self.folder / name) != expected:
                raise ValueError(f"Artifact checksum mismatch: {name}")
        for key in ["threshold", "image_weight"]:
            if not math.isfinite(self.manifest[key]) or not 0 <= self.manifest[key] <= 1:
                raise ValueError(f"Invalid {key}")
        networks = []
        for name in ["resnet50", "densenet121"]:
            checkpoint = torch.load(
                self.folder / f"{name}.pt", map_location="cpu", weights_only=True
            )
            if (
                checkpoint["preprocess"] != PREPROCESS
                or checkpoint["cohort_manifest"] != self.manifest["cohort_manifest"]
            ):
                raise ValueError("Checkpoint preprocessing or cohort mismatch")
            model = build_model(name)
            model.load_state_dict(checkpoint["state_dict"])
            networks.append(model.eval())
        self.encoder = ImageEnsemble(*networks).eval()
        self.text_model = joblib.load(self.folder / "text_model.joblib")
        if list(self.text_model.classifier.classes_) != [0, 1]:
            raise ValueError("Unexpected text class order")
        self.lock = threading.RLock()

    def text(self, indication, comparison=""):
        return TextInput(indication, comparison)

    def prepare_image(self, image):
        return letterbox(image).convert("RGB")

    def image_tensor(self, image):
        return image_transforms()(self.prepare_image(image))

    def image_context(self, tensor):
        return self.encoder(tensor)

    def score_context(self, context, text):
        frame = pd.DataFrame([{"indication": text.indication, "comparison": text.comparison}])
        text_p = float(self.text_model.predict_proba(frame)[0, 1])
        alpha = self.manifest["image_weight"]
        return alpha * context + (1 - alpha) * text_p

    def score_tensor(self, tensor, text):
        return self.score_context(self.image_context(tensor), text)

    def predict(self, image, indication, comparison=""):
        with self.lock, torch.inference_mode():
            p = float(
                self.score_tensor(
                    self.image_tensor(image).unsqueeze(0), self.text(indication, comparison)
                )[0]
            )
        cutoff = self.manifest["threshold"]
        tier, reason = get_priority(p, cutoff)
        return {
            "probability": p,
            "predicted_label": "proxy_abnormal" if p >= cutoff else "proxy_negative",
            "threshold": cutoff,
            "priority_tier": tier,
            "priority_reason": reason,
            "model_version": self.manifest["model_version"],
            "calibrated": False,
        }
