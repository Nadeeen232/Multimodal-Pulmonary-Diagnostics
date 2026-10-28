"""Reviewed label/text rules ported from the team classification notebook.

No notebook execution, Colab setup or deployment side effects.
The team cohort differs from the original project cohort.
"""

import re

import numpy as np
import pandas as pd

SPLIT_RATIOS = (0.70, 0.15, 0.15)
SEED = 42
USE_INDICATION = True
MASK_LABEL_TERMS = True

LABEL_GROUPS = {
    "cardiomegaly": ["cardiomegaly"],
    "effusion": ["pleural effusion", "effusion"],
    "atelectasis": ["atelectasis", "volume loss"],
    "airspace_opacity": ["opacity", "airspace disease", "infiltrate", "consolidation", "pneumonia"],
    "nodule_mass": ["nodule", "mass", "cavitation", "cyst"],
    "edema_congestion": ["pulmonary edema", "pulmonary congestion", "heart failure", "edema"],
    "pneumothorax": ["pneumothorax", "hydropneumothorax", "hemopneumothorax"],
    "emphysema_copd": ["emphysema", "chronic obstructive", "bullous emphysema"],
    "fibrosis": ["pulmonary fibrosis", "interstitial lung disease", "fibrosis"],
    "granuloma": ["calcified granuloma", "granulomatous disease", "granuloma"],
    "scar_thickening": ["cicatrix", "thickening"],
    "vascular_calc": ["calcinosis", "atherosclerosis", "aortic aneurysm"],
    "fracture": ["fracture", "fractures, bone"],
    "spine_degen": [
        "scoliosis",
        "osteophyte",
        "spondylosis",
        "kyphosis",
        "spinal fusion",
        "arthritis",
    ],
    "devices": [
        "catheter",
        "surgical instrument",
        "medical device",
        "stent",
        "foreign body",
        "suture",
    ],
    "hernia": ["hernia"],
    "diaphragm": ["diaphragmatic eventration"],
}

LABEL_EXCLUDE = {"effusion": ["pericardial"]}

QUALITY_TAG = "technical quality of image unsatisfactory"

CORE_FLAGS = [
    "cardiomegaly",
    "effusion",
    "atelectasis",
    "airspace_opacity",
    "nodule_mass",
    "edema_congestion",
    "pneumothorax",
    "emphysema_copd",
    "fibrosis",
    "fracture",
]

CHRONIC_CORE = ["nodule_mass", "emphysema_copd", "fibrosis"]

NO_ACUTE_PATTERN = (
    r"\b(?:no acute (?:cardiopulmonary |pulmonary |chest )?"
    r"(?:abnormalit\w*|disease|findings?|process)|negative for acute)"
)

MASK_RE = re.compile(r"\bX{2,}\b", re.IGNORECASE)

XRAY_MASK_RE = re.compile(r"\b[xX]\s*[-–—]\s*[X]{2,}\b")

NUMBERED_BULLET_RE = re.compile(r"(?:(?<=^)|(?<=[.!?]\s))\d{1,2}[.)]\s+(?=[A-Za-z])")

MULTI_DOT_RE = re.compile(r"\.{2,}")

SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,;:!?])")

REPEATED_PUNCT_RE = re.compile(r"([,;:!?])(?:\s*[,;:!?])+")


def as_text(value) -> str:
    """Return a safe string for a possibly missing CSV value."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def split_medical_sentences(text: str) -> list:
    """Split at sentence-ending punctuation followed by whitespace (decimals like 2.5 are preserved)."""
    text = text.replace("\r", " ").replace("\n", " ")
    text = NUMBERED_BULLET_RE.sub("", text)
    text = XRAY_MASK_RE.sub("x-ray", text)
    text = MULTI_DOT_RE.sub(".", text)
    text = SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    text = REPEATED_PUNCT_RE.sub(r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [part.strip() for part in parts if part.strip()]


def clean_section(value) -> dict:
    """Conservative cleanup. Mask tokens are preserved as [MASK] and counted for audit."""
    raw = as_text(value)
    if not raw:
        return {
            "raw": raw,
            "clean": "",
            "mask_count": 0,
            "sentence_count_before": 0,
            "sentence_count_after": 0,
            "sentences_dropped": 0,
            "changed": False,
        }

    sentences = split_medical_sentences(raw)
    output, mask_count, dropped = [], 0, 0
    for sentence in sentences:
        mask_count += len(MASK_RE.findall(sentence))
        sentence = MASK_RE.sub("[MASK]", sentence)
        sentence = re.sub(r"\s+", " ", sentence).strip()
        sentence = re.sub(r"\s+([.,;:!?])", r"\1", sentence)
        sentence = re.sub(r"([,;:])\s*([,;:])+", r"\1", sentence)
        sentence = sentence.strip(" ;,")
        alpha = re.sub(r"[\W_]+", "", sentence, flags=re.UNICODE)
        if not alpha:  # drop only empty / punctuation-only fragments
            dropped += 1
            continue
        if sentence[-1] not in ".!?":
            sentence += "."
        sentence = sentence[0].upper() + sentence[1:]
        output.append(sentence)

    cleaned = " ".join(output).strip()
    return {
        "raw": raw,
        "clean": cleaned,
        "mask_count": mask_count,
        "sentence_count_before": len(sentences),
        "sentence_count_after": len(output),
        "sentences_dropped": dropped,
        "changed": cleaned != raw,
    }


def split_tags(value) -> list:
    return [item.strip().lower() for item in re.split(r"[;|]", as_text(value)) if item.strip()]


def derive_labels(frame: pd.DataFrame) -> pd.DataFrame:
    empty = pd.Series([[] for _ in range(len(frame))], index=frame.index)
    problems = frame["Problems"].map(split_tags) if "Problems" in frame else empty
    mesh = frame["MeSH"].map(split_tags) if "MeSH" in frame else empty
    combined = [set(p + m) for p, m in zip(problems, mesh, strict=True)]

    frame["is_normal_tag"] = [int("normal" in tags and len(tags) == 1) for tags in problems]
    frame["label_unknown"] = [int("no indexing" in tags) for tags in problems]
    frame["poor_quality"] = [int(QUALITY_TAG in tags) for tags in problems]

    for group, terms in LABEL_GROUPS.items():
        banned = LABEL_EXCLUDE.get(group, [])
        frame[f"lbl_{group}"] = [
            int(
                any(
                    any(term in tag for term in terms) and not any(b in tag for b in banned)
                    for tag in tags
                )
            )
            for tags in combined
        ]
    return frame


def iterative_multilabel_split(y: np.ndarray, ratios=SPLIT_RATIOS, seed=SEED) -> np.ndarray:
    """Assign studies to train/val/test one at a time, always handling the RAREST label first."""
    y = np.asarray(y, dtype=np.int8)
    n, n_labels = y.shape
    if n == 0:
        return np.array([], dtype=object)
    if n < 3:
        raise ValueError("Need at least 3 curated studies to create train/val/test splits.")

    ratios = np.asarray(ratios, dtype=float)
    ratios = ratios / ratios.sum()
    rng = np.random.RandomState(seed)
    desired_sizes = ratios * n
    desired_pos = ratios[:, None] * y.sum(axis=0)[None, :]
    assignment = np.full(n, -1, dtype=int)
    remaining = np.ones(n, dtype=bool)

    while remaining.any():
        rem_pos = y[remaining].sum(axis=0)
        positive_labels = np.flatnonzero(rem_pos > 0)
        if len(positive_labels) == 0:
            break
        label = positive_labels[np.argmin(rem_pos[positive_labels])]
        candidates = np.flatnonzero(remaining & (y[:, label] > 0))
        rng.shuffle(candidates)
        for idx in candidates:
            if not remaining[idx]:
                continue
            demand = desired_pos[:, label]
            best_splits = np.flatnonzero(demand == demand.max())
            if len(best_splits) > 1:
                size_demand = desired_sizes[best_splits]
                best_splits = best_splits[size_demand == size_demand.max()]
            split_id = int(rng.choice(best_splits))
            assignment[idx] = split_id
            remaining[idx] = False
            desired_pos[split_id] = np.maximum(0, desired_pos[split_id] - y[idx])
            desired_sizes[split_id] = max(0, desired_sizes[split_id] - 1)

    for idx in np.flatnonzero(remaining):
        split_id = int(np.argmax(desired_sizes + rng.rand(3) * 1e-6))
        assignment[idx] = split_id
        desired_sizes[split_id] = max(0, desired_sizes[split_id] - 1)

    names = np.array(["train", "val", "test"], dtype=object)
    return names[assignment]


TEXT_ABNORMAL_TERMS = [
    "mediastinal widening",
    "widened mediastinum",
    "cardiomegaly",
    "enlarged heart",
    "opacity",
    "opacities",
    "infiltrate",
    "infiltration",
    "airspace disease",
    "mass",
    "nodule",
    "lung lesion",
    "edema",
    "pulmonary edema",
    "consolidation",
    "pneumonia",
    "atelectasis",
    "pneumothorax",
    "pleural effusion",
    "effusion",
    "pleural thickening",
    "pleural disease",
    "fracture",
    "emphysema",
    "fibrosis",
    "hyperinflation",
    "hyperdistention",
    "pulmonary disease, chronic obstructive",
    "copd",
    "bronchiectasis",
    "tuberculosis",
    "lymphadenopathy",
    "granuloma",
    "calcified granuloma",
    "scarring",
    "cicatrix",
    "scoliosis",
    "osteophyte",
    "spondylosis",
    "calcification",
    "hernia",
    "hypoinflation",
    "degenerative change",
    "degenerative disease",
    "atherosclerosis",
]

TEXT_NORMAL_PATTERNS = [
    r"\bnormal\b",
    r"\bnormal study\b",
    r"\bnegative study\b",
    r"\bno finding(s)?\b",
    r"\bno acute cardiopulmonary abnormality\b",
    r"\bno acute cardiopulmonary disease\b",
    r"\bno acute disease\b",
]


def _term_pattern(term):
    escaped = re.escape(term).replace(r"\ ", r"\s+")
    return re.compile(rf"(?<!\w){escaped}(?!\w)", flags=re.IGNORECASE)


_TEXT_MASK_PATTERNS = [_term_pattern(t) for t in sorted(TEXT_ABNORMAL_TERMS, key=len, reverse=True)]

_TEXT_NORMAL_REGEXES = [re.compile(p, flags=re.IGNORECASE) for p in TEXT_NORMAL_PATTERNS]


def t29_clean_text(value):
    """Lower-case, drop HTML / anonymisation masks (XXXX), tidy spaces. Negations (no, not, without) are kept."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    value = str(value).lower()
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"[\r\n\t]+", " ", value)
    value = re.sub(r"\b(?:x{2,}|<redacted>)\b", " ", value)
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"\s+([,.;:])", r"\1", value)
    return value.strip()


def t29_remove_label_terms(text):
    for pattern in _TEXT_MASK_PATTERNS:
        text = pattern.sub(" ", text)
    for pattern in _TEXT_NORMAL_REGEXES:
        text = pattern.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def make_report_text(row):
    """Leakage-aware text input (Team 29 technique): indication + comparison only.

    Findings, impression and Problems/MeSH are excluded (the target was built from tags, and the
    findings/impression can state the diagnosis). Words the label rules were built from are removed.
    """
    if not USE_INDICATION:
        return "Indication: not provided"
    parts = [t29_clean_text(row.get("indication")), t29_clean_text(row.get("comparison"))]
    if MASK_LABEL_TERMS:
        parts = [t29_remove_label_terms(p) for p in parts]
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def text_meta_features(frame):
    """Small numeric features about the indication / comparison fields (Team 29)."""
    ind = frame["indication_clean_t29"].fillna("").astype(str)
    cmp_ = frame["comparison_clean_t29"].fillna("").astype(str)
    return np.column_stack(
        [
            np.log1p(ind.str.split().str.len().fillna(0).astype(float)),
            np.log1p(cmp_.str.split().str.len().fillna(0).astype(float)),
            ind.str.strip().ne("").astype(float),
            cmp_.str.strip().ne("").astype(float),
            cmp_.str.strip().str.strip(".").isin(["", "none"]).astype(float),
        ]
    )
