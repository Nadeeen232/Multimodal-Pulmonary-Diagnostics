"""Offline adapter for the final Team29 FLAN-T5 report candidate.

Mirrors notebooks/Team29_Report_Generation_Final.ipynb (section 09): ResNet18 visual
tokens, IU-classifier finding tags on CheXpert-only XRV pooled features, then
forced finding sentences. This experimental generator does not replace the
verified rule-based summary.
"""

import json
import re
from pathlib import Path

import numpy as np
import torch
from torch import nn

from pulmonary.common import file_hash

RESNET_DIMENSION = 512
XRV_DIMENSION = 1024
# Contradicting statements for cardiomegaly are phrased about size, not negated.
CARDIAC_NORMAL = (
    r"(heart|cardiac|cardiomediastinal|mediastinal)[^.]*(normal|unremarkable|within normal limits)"
)
REQUIRED_FILES = {
    "config.json",
    "visual_adapter.pt",
    "resnet18_encoder.pt",
    "iu_tag_classifier.pt",
    "text_model/config.json",
    "text_model/model.safetensors",
    "tokenizer/tokenizer_config.json",
    "tokenizer/tokenizer.json",
}


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


class FindingTagger:
    """IU finding classifier, prompt tags and forced findings from the exported config."""

    def __init__(self, config, classifier):
        tags, forced = config["tags"], config["forced_findings"]
        if tags.get("mode") != "iu_classifier":
            raise ValueError("Expected the final IU-classifier export, not raw XRV score tags")
        if str(tags.get("xrv_weights", "")).endswith("-all"):
            raise ValueError("XRV weights trained on OpenI overlap IU X-Ray; use CheXpert-only")
        self.names = list(tags["names"])
        self.thresholds = [float(t) for t in tags["thresholds"]]
        self.patterns = {k: tags["patterns"][k] for k in self.names}
        self.negation = tags["negation"]
        self.max_tags = int(tags["max_tags"])
        self.forced_thresholds = forced["thresholds"]
        self.sentences = {k: tuple(v) for k, v in forced["sentences"].items()}
        weight, bias = classifier["weight"].float(), classifier["bias"].float()
        self.mu = classifier["mu"].float().numpy()
        self.sd = classifier["sd"].float().numpy()
        count = len(self.names)
        if (
            len(self.thresholds) != count
            or weight.shape != (count, XRV_DIMENSION)
            or bias.shape != (count,)
            or self.mu.shape != (XRV_DIMENSION,)
            or self.sd.shape != (XRV_DIMENSION,)
            or not set(self.forced_thresholds) <= set(self.names)
        ):
            raise ValueError("IU tag classifier does not match the exported finding names")
        self.weight, self.bias = weight, bias

    def finding_probs(self, xrv_pooled):
        pooled = np.asarray(xrv_pooled, dtype=np.float32)
        if pooled.ndim != 2 or pooled.shape[1] != XRV_DIMENSION or not np.isfinite(pooled).all():
            raise ValueError("Expected finite 1024-dimensional pooled XRV vectors, one per image")
        x = ((pooled.mean(axis=0) - self.mu) / self.sd).astype(np.float32)[None]
        with torch.no_grad():
            logits = nn.functional.linear(torch.from_numpy(x), self.weight, self.bias)
        return torch.sigmoid(logits)[0].numpy()

    def make_tags(self, probs):
        pairs = zip(self.names, self.thresholds, strict=True)
        selected = sorted(
            [(probs[j] - t, name) for j, (name, t) in enumerate(pairs) if probs[j] >= t],
            reverse=True,
        )
        return ", ".join(name for _, name in selected[: self.max_tags]) or "none"

    def build_text(self, indication, probs):
        note = " ".join(re.sub(r"\b[xX]{2,}\b", " ", str(indication)).split())
        return (
            "generate chest x-ray report. indication: "
            + (note or "not provided")
            + ". image findings: "
            + self.make_tags(probs)
        )

    def mentions(self, text, finding):
        for sentence in re.split(r"[.;]", str(text).lower()):
            for match in re.finditer(self.patterns[finding], sentence):
                if not re.search(self.negation, sentence[: match.start()]):
                    return True
        return False

    def contradicts(self, sentence, finding):
        if finding == "cardiomegaly":
            return bool(re.search(CARDIAC_NORMAL, sentence))
        return any(
            re.search(self.negation, sentence[: m.start()])
            for m in re.finditer(self.patterns[finding], sentence)
        )

    def enforce(self, report, probs):
        """Add a fixed sentence for confident, unmentioned findings; drop contradictions."""

        def split(text):
            return [s.strip() for s in re.split(r"(?<=\.)\s+", text) if s.strip()]

        text = str(report).lower().strip()
        cut = text.find("impression:")
        findings = (text[:cut] if cut >= 0 else text).replace("findings:", "", 1).strip()
        impression = text[cut + len("impression:") :].strip() if cut >= 0 else ""
        findings, impression, added = split(findings), split(impression), []
        for j, name in enumerate(self.names):
            threshold = self.forced_thresholds.get(name)
            if (
                threshold is None
                or probs[j] < threshold
                or name not in self.sentences
                or self.mentions(text, name)
            ):
                continue
            findings = [s for s in findings if not self.contradicts(s, name)]
            findings.append(self.sentences[name][0])
            impression = [self.sentences[name][1]] + [
                s for s in impression if not self.contradicts(s, name)
            ]
            added.append(name)
        return "findings: " + " ".join(findings) + " impression: " + " ".join(impression), added


class LearnedReportGenerator:
    """Load local weights without executing supplied notebook or inference scripts."""

    def __init__(self, folder, device="cpu"):
        self.folder = Path(folder)
        self.config = json.loads((self.folder / "config.json").read_text())
        manifest = self.folder / "checksums.json"
        if not manifest.exists():
            raise ValueError("Create and review checksums.json for this imported export first")
        expected_files = json.loads(manifest.read_text())
        if not REQUIRED_FILES.issubset(expected_files):
            raise ValueError("Incomplete report generator checksum manifest")
        for name, digest in expected_files.items():
            path = (self.folder / name).resolve()
            if not path.is_relative_to(self.folder.resolve()) or file_hash(path) != digest:
                raise ValueError(f"Invalid report generator artifact: {name}")
        if not 1 <= self.config["visual_tokens"] <= 16:
            raise ValueError("Unsupported visual token count")
        if not 1 <= self.config["max_input"] <= 512 or not 1 <= self.config["max_target"] <= 512:
            raise ValueError("Unsupported report token limits")
        classifier = torch.load(
            self.folder / "iu_tag_classifier.pt", map_location="cpu", weights_only=True
        )
        self.tagger = FindingTagger(self.config, classifier)
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

    def build_text(self, indication, xrv_pooled):
        return self.tagger.build_text(indication, self.tagger.finding_probs(xrv_pooled))

    @torch.inference_mode()
    def generate_features(self, vectors, view_ids, indication, xrv_pooled, forced=True):
        vectors = np.asarray(vectors, dtype=np.float32)
        views = np.asarray(view_ids)
        if vectors.ndim != 2 or vectors.shape[1] != RESNET_DIMENSION or not 1 <= len(vectors) <= 8:
            raise ValueError("Expected one to eight 512-dimensional ResNet18 feature vectors")
        if (
            not np.isfinite(vectors).all()
            or views.shape != (len(vectors),)
            or not np.isin(views, [1, 2, 3]).all()
        ):
            raise ValueError("Invalid visual features or view identifiers")
        if len(xrv_pooled) != len(vectors):
            raise ValueError("Each image needs one pooled XRV feature vector")
        probs = self.tagger.finding_probs(xrv_pooled)
        prompt = self.tagger.build_text(indication, probs)
        tokens = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=self.config["max_input"]
        ).to(self.device)
        embeddings, mask = self.net.encode(
            torch.tensor(vectors[None], device=self.device),
            torch.tensor(views[None], dtype=torch.long, device=self.device),
            torch.ones(1, len(vectors), dtype=torch.long, device=self.device),
            tokens["input_ids"],
            tokens["attention_mask"],
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
        draft = self.tokenizer.decode(output[0], skip_special_tokens=True)
        report, added = self.tagger.enforce(draft, probs) if forced else (draft, [])
        return {
            "report": report,
            "draft_report": draft,
            "forced_findings": added,
            "finding_probabilities": {
                n: float(p) for n, p in zip(self.tagger.names, probs, strict=True)
            },
            "method": "Team29 FLAN-T5-small image-conditioned draft with IU-classifier tags",
            "status": "experimental; clinician verification is required",
            "prompt": prompt,
        }
