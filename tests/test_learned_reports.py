"""The final Team29 report adapter must reproduce the notebook's tags and forced findings."""

import json

import numpy as np
import pytest
import torch

from pulmonary.common import file_hash
from pulmonary.learned_reports import FindingTagger, LearnedReportGenerator

NAMES = ["cardiomegaly", "atelectasis", "opacity", "effusion", "emphysema", "degenerative spine"]
NEGATION = r"\b(no|not|without|negative|free of|clear of|resolved|resolution of)\b"
PATTERNS = {
    "cardiomegaly": r"cardiomegaly|enlarged heart|heart is enlarged|cardiac enlargement|enlarged cardiac",
    "atelectasis": r"atelecta",
    "opacity": r"opacit|consolidation|infiltrat|airspace disease",
    "effusion": r"effusion",
    "emphysema": r"emphysema|hyperinflat|hyperexpan|copd",
    "degenerative spine": r"degenerative|spondylo|osteophyte|spurring|arthriti",
}
SENTENCES = {
    "cardiomegaly": ["the heart is enlarged.", "cardiomegaly."],
    "atelectasis": ["there is mild atelectasis.", "mild atelectasis."],
    "opacity": ["there is a focal airspace opacity.", "focal airspace opacity."],
    "effusion": ["there is a small pleural effusion.", "small pleural effusion."],
    "emphysema": ["the lungs are hyperexpanded, consistent with emphysema.", "emphysema."],
    "degenerative spine": [
        "there are degenerative changes of the thoracic spine.",
        "degenerative changes of the spine.",
    ],
}


def export_config(**tags):
    return {
        "visual_tokens": 4,
        "max_input": 128,
        "max_target": 8,
        "tags": {
            "mode": "iu_classifier",
            "xrv_weights": "densenet121-res224-chex",
            "names": NAMES,
            "thresholds": [0.63, 0.54, 0.5, 0.72, 0.74, 0.45],
            "max_tags": 3,
            "patterns": PATTERNS,
            "negation": NEGATION,
            **tags,
        },
        "forced_findings": {
            "precision_target": 0.6,
            "thresholds": {
                "cardiomegaly": 0.847,
                "atelectasis": None,
                "opacity": 0.725,
                "effusion": 0.943,
                "emphysema": 0.927,
                "degenerative spine": None,
            },
            "sentences": SENTENCES,
        },
        "decoding": {"num_beams": 2, "no_repeat_ngram_size": 3, "repetition_penalty": 1.2},
    }


def classifier(bias=None):
    """Zero weights and unit scaling make each probability sigmoid(bias)."""
    return {
        "weight": torch.zeros(len(NAMES), 1024),
        "bias": torch.zeros(len(NAMES)) if bias is None else torch.tensor(bias),
        "mu": torch.zeros(1024),
        "sd": torch.ones(1024),
    }


def probs(**values):
    return np.array([values.get(n.replace(" ", "_"), 0.1) for n in NAMES], dtype=np.float32)


@pytest.mark.parametrize(
    ("draft", "expected"),
    [
        (
            "findings: the cardiomediastinal silhouette is within normal limits for size and contour. the lungs are free of focal airspace disease, pleural effusion, or pneumothorax. there is no focal consolidation, effusion or focal consolidation. impression: 1. no acute cardiopulmonary abnormality.",
            "findings: the cardiomediastinal silhouette is within normal limits for size and contour. there is a focal airspace opacity. impression: focal airspace opacity. 1. no acute cardiopulmonary abnormality.",
        ),
        (
            "findings: the heart size and mediastinal contours are within normal limits. there is no pneumothorax or pleural effusion. the lungs are otherwise clear. impression: no acute cardiopulmonary abnormality.",
            "findings: the heart size and mediastinal contours are within normal limits. there is no pneumothorax or pleural effusion. the lungs are otherwise clear. there is a focal airspace opacity. impression: focal airspace opacity. no acute cardiopulmonary abnormality.",
        ),
        (
            "findings: the heart size and mediastinal contours are within normal limits. there is no pneumothorax or large pleural effusion. there are no focal airspace consolidation, effusions, or atelectasis. impression: no acute cardiopulmonary abnormality.",
            "findings: the heart size and mediastinal contours are within normal limits. there is no pneumothorax or large pleural effusion. there is a focal airspace opacity. impression: focal airspace opacity. no acute cardiopulmonary abnormality.",
        ),
    ],
)
def test_forced_opacity_matches_notebook_validation_examples(draft, expected):
    """Draft/hybrid pairs printed by the notebook's section 07b on validation."""
    tagger = FindingTagger(export_config(), classifier())
    report, added = tagger.enforce(draft, probs(opacity=0.9))
    assert report == expected
    assert added == ["opacity"]


def test_forced_cardiomegaly_replaces_contradicting_normal_heart():
    tagger = FindingTagger(export_config(), classifier())
    draft = "findings: the heart size is normal. the lungs are clear. impression: the heart is normal in size."
    report, added = tagger.enforce(draft, probs(cardiomegaly=0.9))
    assert added == ["cardiomegaly"]
    assert report == (
        "findings: the lungs are clear. the heart is enlarged. impression: cardiomegaly."
    )


def test_findings_are_not_forced_when_stated_below_threshold_or_never_forced():
    tagger = FindingTagger(export_config(), classifier())
    stated = "findings: there is a right basilar opacity. impression: right basilar opacity."
    assert tagger.enforce(stated, probs(opacity=0.99)) == (stated, [])
    plain = "findings: the lungs are clear. impression: no acute cardiopulmonary abnormality."
    assert tagger.enforce(plain, probs(opacity=0.7))[1] == []
    assert tagger.enforce(plain, probs(atelectasis=0.99, degenerative_spine=0.99))[1] == []


def test_prompt_tags_rank_by_margin_and_strip_deidentification():
    # sigmoid: .88, .01, .55, .95, .77, .27; four pass, emphysema has the smallest margin
    bias = [2.0, -5.0, 0.2, 3.0, 1.2, -1.0]
    tagger = FindingTagger(export_config(), classifier(bias))
    p = tagger.finding_probs(np.zeros((2, 1024), dtype=np.float32))
    assert p == pytest.approx(torch.sigmoid(torch.tensor(bias)).numpy(), abs=1e-6)
    assert tagger.build_text("XXXX preop  chest", p) == (
        "generate chest x-ray report. indication: preop chest. "
        "image findings: cardiomegaly, effusion, opacity"
    )
    none = FindingTagger(export_config(), classifier([-9.0] * len(NAMES)))
    assert none.build_text("", none.finding_probs(np.zeros((1, 1024)))) == (
        "generate chest x-ray report. indication: not provided. image findings: none"
    )


@pytest.mark.parametrize(
    ("tags", "message"),
    [
        ({"mode": "xrv_q0.88"}, "IU-classifier"),
        ({"xrv_weights": "densenet121-res224-all"}, "OpenI"),
        ({"thresholds": [0.5]}, "does not match"),
    ],
)
def test_tagger_rejects_superseded_or_inconsistent_exports(tags, message):
    with pytest.raises(ValueError, match=message):
        FindingTagger(export_config(**tags), classifier())


def test_tagger_rejects_wrong_feature_shape():
    tagger = FindingTagger(export_config(), classifier())
    with pytest.raises(ValueError, match="1024"):
        tagger.finding_probs(np.zeros((1, 512), dtype=np.float32))


def write_export(folder):
    """Tiny random FLAN-T5-shaped export in the notebook's section 09 layout."""
    pytest.importorskip("transformers")
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, T5Config, T5ForConditionalGeneration

    from pulmonary.learned_reports import VisualReportModel

    words = ["<pad>", "</s>", "<unk>", "generate", "chest", "report", "the", "lungs", "clear"]
    backend = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", eos_token="</s>", unk_token="<unk>"
    ).save_pretrained(folder / "tokenizer")
    torch.manual_seed(0)
    T5ForConditionalGeneration(
        T5Config(
            vocab_size=len(words),
            d_model=16,
            d_ff=32,
            d_kv=8,
            num_layers=1,
            num_heads=2,
            decoder_start_token_id=0,
            pad_token_id=0,
            eos_token_id=1,
        )
    ).save_pretrained(folder / "text_model", safe_serialization=True)
    net = VisualReportModel(folder / "text_model", 4)
    adapter = {k: v for k, v in net.state_dict().items() if not k.startswith("text_model.")}
    torch.save(adapter, folder / "visual_adapter.pt")
    torch.save(classifier([-3.0, -3.0, 3.0, -3.0, -3.0, -3.0]), folder / "iu_tag_classifier.pt")
    (folder / "resnet18_encoder.pt").write_bytes(b"encoder placeholder")
    (folder / "config.json").write_text(json.dumps(export_config()))
    files = [p for p in folder.rglob("*") if p.is_file()]
    checksums = {p.relative_to(folder).as_posix(): file_hash(p) for p in files}
    (folder / "checksums.json").write_text(json.dumps(checksums))


def test_generator_loads_final_export_and_forces_confident_finding(tmp_path):
    """Random weights verify software parity only; no report-quality claim."""
    write_export(tmp_path)
    generator = LearnedReportGenerator(tmp_path)
    result = generator.generate_features(
        np.zeros((2, 512), dtype=np.float32), [1, 2], "XXXX cough", np.zeros((2, 1024))
    )
    assert result["prompt"] == (
        "generate chest x-ray report. indication: cough. image findings: opacity"
    )
    assert result["forced_findings"] == ["opacity"]
    assert result["report"].endswith("impression: focal airspace opacity.")
    assert "there is a focal airspace opacity." in result["report"]
    assert set(result["finding_probabilities"]) == set(NAMES)
    plain = generator.generate_features(
        np.zeros((1, 512)), [1], "cough", np.zeros((1, 1024)), forced=False
    )
    assert plain["report"] == plain["draft_report"] and plain["forced_findings"] == []


def test_generator_rejects_tampered_classifier(tmp_path):
    write_export(tmp_path)
    torch.save(classifier(), tmp_path / "iu_tag_classifier.pt")
    with pytest.raises(ValueError, match="iu_tag_classifier.pt"):
        LearnedReportGenerator(tmp_path)


def test_generator_rejects_mismatched_feature_counts(tmp_path):
    write_export(tmp_path)
    generator = LearnedReportGenerator(tmp_path)
    with pytest.raises(ValueError, match="pooled XRV"):
        generator.generate_features(np.zeros((2, 512)), [1, 2], "", np.zeros((1, 1024)))
