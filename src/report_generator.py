"""Deterministic clinical-format reports describing measured model evidence."""

import math


def generate_report(probability, priority, gradcam_region, top_text_tokens):
    """Return five plain-text sections from measured attribution evidence.

    ``priority`` is the existing validation-threshold queue tier. Token dictionaries
    contain ``token`` and signed ``contribution`` (score change on removal).
    Strings are accepted as tokens with unspecified attribution direction.
    """
    probability = float(probability)
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Probability must be finite and between zero and one")
    if priority not in {"lower_review", "review", "higher_review"}:
        raise ValueError("Unknown review priority")
    predicted_class = "Normal" if priority == "lower_review" else "Abnormal"
    relation = "below" if priority == "lower_review" else "at or above"
    impression = (
        f"The AI model assigns an abnormality score of {probability:.1%}, "
        f"{relation} the configured decision threshold. "
        "This score is uncalibrated and is not a probability of disease."
    )
    region = " ".join(str(gradcam_region or "unavailable").split())[:160]
    region = (
        region.replace(" (patient laterality unverified)", "")
        .replace("image-left", "left")
        .replace("image-right", "right")
        .replace("image region", "area of the radiograph")
    )
    if "unavailable" in region:
        visual = "No interpretable positive Grad-CAM attention region was available."
    elif region.startswith("diffuse"):
        visual = (
            "Model attention is spread across the radiograph without a dominant vertical region."
        )
    else:
        visual = f"Model attention primarily focuses on the {region}."
    visual += " Attention does not establish a lesion or its absence; image position does not verify anatomy or patient laterality."
    tokens = {"raises": [], "lowers": [], "unchanged": [], "unspecified": []}
    for item in (top_text_tokens or [])[:5]:
        token = item.get("token", "") if isinstance(item, dict) else str(item)
        token = " ".join(token.split())[:60]
        if not token:
            continue
        direction = "unspecified"
        if isinstance(item, dict) and "contribution" in item:
            delta = float(item["contribution"])
            if not math.isfinite(delta):
                raise ValueError("Token contribution must be finite")
            direction = "raises" if delta > 0 else "lowers" if delta < 0 else "unchanged"
        tokens[direction].append(f"'{token}'")
    context = []
    for direction, wording in [
        ("raises", "were associated with a higher abnormality score"),
        ("lowers", "were associated with a lower abnormality score"),
        ("unchanged", "had no measured effect on the score"),
        ("unspecified", "were included without a measured attribution direction"),
    ]:
        if tokens[direction]:
            context.append(f"Clinical indication terms {', '.join(tokens[direction])} {wording}.")
    text_evidence = (
        " ".join(context) if context else "No informative text attribution was available."
    )
    text_evidence += (
        " These associations describe model sensitivity, not the clinical significance of symptoms."
    )
    recommendation = (
        "Prioritized radiologist review. Queue priority does not establish clinical urgency."
        if priority == "higher_review"
        else "Radiologist review recommended."
        if priority == "review"
        else "Routine radiologist review. A Normal prediction does not exclude disease."
    )
    return (
        f"Predicted Class: {predicted_class}\n\n"
        f"Impression:\n{impression}\n\n"
        f"Visual Findings:\n{visual}\n\n"
        f"Clinical Context:\n{text_evidence}\n\n"
        f"Recommendation:\n{recommendation} Clinician verification is required."
    )
