# Review priority and decision policy

Current model: **0.2.0-team29-47e281fc**. Validation balanced accuracy selected
the binary threshold **0.49**. The old 10:1 cost rule is not active in this bundle.
Scores estimate agreement with weak labels; they are uncalibrated.

| Tier | Score range | Interpretation |
| --- | --- | --- |
| Lower review | Below 0.49 | Below the model decision threshold; not disease exclusion |
| Review | At least 0.49, below 0.745 | Meets the model threshold |
| Higher review | At least 0.745 | Meets an illustrative higher-score cutoff |

The upper cutoff is threshold + 0.5 * (1 - threshold). This is illustrative
queue ordering, not validated medical urgency. Symptoms do not trigger an
independent urgency rule. Source: src/pulmonary/priority.py.

Current fusion precision is 83.45% and recall 75.32% on the study test cohort.
The raw reliability plot is descriptive; no calibrator has been applied. Clinical
deployment would need stronger labels, external validation and a calibration design.

## Historical v1 policy

The earlier v1 experiment minimized 10 * FN + FP on validation, with a different
cohort and threshold near 0.04686. Its high recall came with many false positives.
Those cost curves remain historical evidence and do not describe the active model.
