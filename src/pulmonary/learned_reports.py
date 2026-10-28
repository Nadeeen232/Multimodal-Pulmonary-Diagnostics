"""Offline adapter for the reviewed Team29 FLAN-T5 report candidate.

This experimental generator does not replace the verified rule-based summary.
Inputs must use the exported encoder's feature and XRV pathology ordering.
"""

import json
import re
from pathlib import Path

import numpy as np
import torch
from torch import nn

from pulmonary.common import file_hash


class VisualReportModel(nn.Module):
    def __init__(self, text_model, visual_tokens=4):
        super().__init__()
        from transformers import AutoModelForSeq2SeqLM

        self.text_model = AutoModelForSeq2SeqLM.from_pretrained(
            str(text_model), local_files_only=True, use_safetensors=True
        )
        self.d = self.text_model.config.d_model
        self.k = visual_tokens
        self.projector = nn.Sequential(
            nn.LayerNorm(512), nn.Linear(512, 512), nn.GELU(), nn.Linear(512, self.k * self.d)
        )
        self.view_embedding = nn.Embedding(4, self.d, padding_idx=0)
        self.visual_norm = nn.LayerNorm(self.d)
        with torch.no_grad():
            scale = (
                self.text_model.get_input_embeddings().weight[:4096].float().square().mean().sqrt()
            )
        self.register_buffer("visual_scale", scale.clone())

    def encode(self, vectors, views, image_mask, input_ids, attention_mask):
        batch, count, _ = vectors.shape
        visual = self.projector(vectors).reshape(batch, count, self.k, self.d)
        visual = visual + self.view_embedding(views).unsqueeze(2)
        visual = self.visual_norm(visual) * self.visual_scale
        visual = visual.reshape(batch, count * self.k, self.d)
        text = self.text_model.get_input_embeddings()(input_ids)
        mask = torch.cat([image_mask.repeat_interleave(self.k, dim=1), attention_mask], dim=1)
        return torch.cat([visual, text], dim=1), mask


class LearnedReportGenerator:
    """Load local weights without executing supplied notebook or inference scripts."""

    def __init__(self, folder, device="cpu"):
        self.folder = Path(folder)
        self.config = json.loads((self.folder / "config.json").read_text())
        manifest = self.folder / "checksums.json"
        if not manifest.exists():
            raise ValueError("Create and review checksums.json for this imported export first")
        expected_files = json.loads(manifest.read_text())
        required = {
            "config.json",
            "visual_adapter.pt",
            "resnet18_encoder.pt",
            "text_model/config.json",
            "text_model/model.safetensors",
            "tokenizer/tokenizer_config.json",
            "tokenizer/tokenizer.json",
        }
        if not required.issubset(expected_files):
            raise ValueError("Incomplete report generator checksum manifest")
        for name, digest in expected_files.items():
            path = (self.folder / name).resolve()
            if not path.is_relative_to(self.folder.resolve()) or file_hash(path) != digest:
                raise ValueError(f"Invalid report generator artifact: {name}")
        if not 1 <= self.config["visual_tokens"] <= 16:
            raise ValueError("Unsupported visual token count")
        if not 1 <= self.config["max_input"] <= 512 or not 1 <= self.config["max_target"] <= 512:
            raise ValueError("Unsupported report token limits")
        from transformers import AutoTokenizer

        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.folder / "tokenizer"), local_files_only=True
        )
        self.net = VisualReportModel(self.folder / "text_model", self.config["visual_tokens"])
        state = torch.load(self.folder / "visual_adapter.pt", map_location="cpu", weights_only=True)
        expected = {k for k in self.net.state_dict() if not k.startswith("text_model.")}
        if set(state) != expected:
            raise ValueError("Adapter keys do not match the reviewed architecture")
        self.net.load_state_dict(state, strict=False)
        self.net.to(self.device).eval()

    def make_tags(self, scores):
        scores = np.asarray(scores, dtype=np.float32)
        count = len(self.config["tags"]["pathologies"])
        if scores.ndim != 2 or scores.shape[1] != count or not np.isfinite(scores).all():
            raise ValueError("Expected finite XRV score vectors in exported pathology order")
        maximum = scores.max(axis=0)
        selected = sorted(
            [
                (maximum[i] - np.float64(t), name.lower().replace("_", " "))
                for i, name, t in self.config["tags"]["rules"]
                if maximum[i] >= np.float64(t)
            ],
            reverse=True,
        )
        return ", ".join(name for _, name in selected[: self.config["tags"]["max_tags"]]) or "none"

    def build_text(self, indication, scores):
        note = " ".join(re.sub(r"\b[xX]{2,}\b", " ", str(indication)).split())
        return (
            "generate chest x-ray report. indication: "
            + (note or "not provided")
            + ". image findings: "
            + self.make_tags(scores)
        )

    @torch.inference_mode()
    def generate_features(self, vectors, view_ids, indication, xrv_scores):
        vectors, scores = (
            np.asarray(vectors, dtype=np.float32),
            np.asarray(xrv_scores, dtype=np.float32),
        )
        views = np.asarray(view_ids)
        if vectors.ndim != 2 or vectors.shape[1] != 512 or not 1 <= len(vectors) <= 8:
            raise ValueError("Expected one to eight 512-dimensional ResNet18 feature vectors")
        if (
            not np.isfinite(vectors).all()
            or views.shape != (len(vectors),)
            or not np.isin(views, [1, 2, 3]).all()
        ):
            raise ValueError("Invalid visual features or view identifiers")
        if len(scores) != len(vectors):
            raise ValueError("Each image needs one pathology-score vector")
        prompt = self.build_text(indication, scores)
        tokens = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=self.config["max_input"]
        ).to(self.device)
        embeddings, mask = self.net.encode(
            torch.tensor(vectors[None], device=self.device),
            torch.tensor(views[None], dtype=torch.long, device=self.device),
            torch.ones(1, len(vectors), dtype=torch.long, device=self.device),
            **tokens,
        )
        decoding = self.config["decoding"]
        output = self.net.text_model.generate(
            inputs_embeds=embeddings,
            attention_mask=mask,
            max_new_tokens=self.config["max_target"],
            num_beams=decoding["num_beams"],
            no_repeat_ngram_size=decoding["no_repeat_ngram_size"],
            repetition_penalty=decoding["repetition_penalty"],
            do_sample=False,
        )
        return {
            "report": self.tokenizer.decode(output[0], skip_special_tokens=True),
            "method": "Team29 FLAN-T5-small image-conditioned draft",
            "status": "experimental; clinician verification is required",
            "prompt": prompt,
        }
