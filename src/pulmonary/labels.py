"""Versioned weak-label mapping migrated from the supplied notebook.

This mapping borrows category names; it is NOT the official CheXpert labeler.
"""

import re

LABEL_VERSION = "indiana-weak-v1"
CORE_TERMS = [
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
]
INCIDENTAL_TERMS = [
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


def term_pattern(term):
    return re.compile(r"(?<!\w)" + re.escape(term).replace(r"\ ", r"\s+") + r"(?!\w)", re.I)


CORE_PATTERNS = [term_pattern(term) for term in CORE_TERMS]


def clean_text(value):
    if value is None:
        return ""
    text = re.sub(r"<[^>]+>", " ", str(value).lower())
    text = re.sub(r"\bx{2,}\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def weak_label(problems, mesh):
    problems, mesh = clean_text(problems), clean_text(mesh)
    core = any(pattern.search(f"{problems} {mesh}") for pattern in CORE_PATTERNS)
    if problems == "normal":
        return "uncertain" if core else "normal"
    return "abnormal" if core else "uncertain"


def model_text(row, fields=("indication",), strip_target_terms=False):
    # Only text available before the current radiology interpretation is eligible.
    if not set(fields).issubset({"indication", "comparison"}):
        raise ValueError("Only indication/comparison are permitted predictor text fields")
    pieces = []
    for field in fields:
        text = clean_text(row.get(field, ""))
        if strip_target_terms:
            for term in CORE_TERMS + INCIDENTAL_TERMS + ["normal", "no findings"]:
                text = term_pattern(term).sub(" ", text)
        if text.strip():
            pieces.append(field + " " + re.sub(r"\s+", " ", text).strip())
    return " ".join(pieces)
