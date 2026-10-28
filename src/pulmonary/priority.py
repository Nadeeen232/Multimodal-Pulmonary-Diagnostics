"""Illustrative review ordering; no symptom rules or medical urgency claims."""

import math


def get_priority(probability, threshold):
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Probability must be finite and in [0, 1]")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be finite and in [0, 1]")
    upper = threshold + 0.5 * (1 - threshold)
    if probability >= upper:
        return (
            "higher_review",
            f"Model score meets the illustrative higher-review cutoff ({upper:.3f}).",
        )
    if probability >= threshold:
        return (
            "review",
            f"Model score meets the validation-selected decision threshold ({threshold:.3f}).",
        )
    return (
        "lower_review",
        f"Model score is below the decision threshold ({threshold:.3f}); abnormalities are not ruled out.",
    )
