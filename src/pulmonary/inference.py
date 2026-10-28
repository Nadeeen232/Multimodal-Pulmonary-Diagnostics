"""Trusted local artifact loading and the shared multimodal forward path."""

import json
import math
import threading
from pathlib import Path

import joblib
import torch

from pulmonary.common import file_hash
from pulmonary.images import PREPROCESS_VERSION, image_tensor
from pulmonary.labels import model_text
from pulmonary.networks import FusionHead, load_encoder
from pulmonary.priority import get_priority


def load_engine(bundle):
    """Dispatch only explicitly supported, versioned artifact formats."""
    manifest = json.loads((Path(bundle) / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("kind") == "team29_late":
        from pulmonary.team_inference import TeamInferenceEngine

        return TeamInferenceEngine(bundle)
    return InferenceEngine(bundle)


class InferenceEngine:
    def __init__(self, bundle):
        self.folder = Path(bundle)
        self.manifest = json.loads((self.folder / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest["preprocessing_version"] != PREPROCESS_VERSION:
            raise ValueError("Preprocessing version mismatch")
        if self.manifest.get("schema_version") != 1:
            raise ValueError("Unsupported bundle schema")
        self.kind = self.manifest["kind"]
        required = (
            {"backbone.pt", "fusion.pt", "vectorizer.joblib", "scaler.joblib"}
            if self.kind == "early"
            else {"backbone.pt", "text.joblib"}
        )
        if self.kind not in {"early", "late"} or set(self.manifest["files"]) != required:
            raise ValueError(
                "Serving bundle must checksum all and only required multimodal artifacts"
            )
        threshold = self.manifest["threshold"]
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Invalid operating threshold")
        if not 32 <= self.manifest["image_size"] <= 1024:
            raise ValueError("Unsupported image size")
        if "indication" not in self.manifest["text_fields"]:
            raise ValueError("Serving requires indication text")
        for name, expected in self.manifest["files"].items():
            path = self.folder / name
            if path.resolve().parent != self.folder.resolve():
                raise ValueError("Invalid artifact path")
            if file_hash(path) != expected:
                raise ValueError(f"Artifact checksum mismatch: {name}")
        self.encoder, checkpoint = load_encoder(self.folder / "backbone.pt")
        if checkpoint["image_size"] != self.manifest["image_size"]:
            raise ValueError("Image size mismatch")
        self.lock = threading.RLock()
        if self.kind == "early":
            self.vectorizer = joblib.load(self.folder / "vectorizer.joblib")
            self.scaler = joblib.load(self.folder / "scaler.joblib")
            checkpoint = torch.load(
                self.folder / "fusion.pt", map_location="cpu", weights_only=True
            )
            self.fusion = FusionHead(
                checkpoint["input_dim"], checkpoint["hidden"], checkpoint["dropout"]
            )
            self.fusion.load_state_dict(checkpoint["state_dict"])
            if checkpoint["input_dim"] != self.encoder.feature_dim + len(
                self.vectorizer.get_feature_names_out()
            ):
                raise ValueError("Image/text dimensions do not match fusion checkpoint")
            self.fusion.eval()
        elif self.kind == "late":
            self.text_model = joblib.load(self.folder / "text.joblib")
            if list(self.text_model.classes_) != [0, 1]:
                raise ValueError("Unexpected text class order")
            if not 0 < self.manifest["image_weight"] < 1:
                raise ValueError("Both modalities must have nonzero weight")
        else:
            raise ValueError("Serving requires a multimodal model")

    def text(self, indication, comparison=""):
        return model_text(
            {"indication": indication, "comparison": comparison},
            self.manifest["text_fields"],
            self.manifest["strip_target_terms"],
        )

    def score_tensor(self, tensor, text):
        return self.score_context(self.image_context(tensor), text)

    def image_context(self, tensor):
        return (
            self.encoder(tensor).softmax(1)[:, 1]
            if self.kind == "late"
            else self.encoder.embed(tensor)
        )

    def score_context(self, image_context, text):
        if self.kind == "late":
            text_p = float(self.text_model.predict_proba([text])[0, 1])
            alpha = self.manifest["image_weight"]
            return alpha * image_context + (1 - alpha) * text_p
        text_features = torch.tensor(
            self.vectorizer.transform([text]).toarray(), dtype=torch.float32
        )
        combined = torch.cat([image_context, text_features], dim=1)
        mean = torch.tensor(self.scaler.mean_, dtype=torch.float32)
        scale = torch.tensor(self.scaler.scale_, dtype=torch.float32)
        return self.fusion((combined - mean) / scale).softmax(1)[:, 1]

    def predict(self, image, indication, comparison=""):
        text = self.text(indication, comparison)
        tensor = image_tensor(image, self.manifest["image_size"]).unsqueeze(0)
        with self.lock, torch.inference_mode():
            probability = float(self.score_tensor(tensor, text)[0])
        threshold = self.manifest["threshold"]
        tier, reason = get_priority(probability, threshold)
        return {
            "probability": probability,
            "predicted_label": "proxy_abnormal" if probability >= threshold else "proxy_negative",
            "threshold": threshold,
            "priority_tier": tier,
            "priority_reason": reason,
            "model_version": self.manifest["model_version"],
            "calibrated": False,
        }
