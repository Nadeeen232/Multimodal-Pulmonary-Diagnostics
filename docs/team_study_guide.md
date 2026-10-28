# Team study and handoff guide - current release

Current handoff: 28 October 2026. Submission: Thursday 29 October 2026; exact time
was not supplied. All primary presentation facts use **0.2.0-team29-47e281fc**.

## Martina and Yousef - data preparation and EDA

Study Data Assets, the Team29 review, `team_classification.prepare` and
`team_rules.py`. Explain the 3,851 reports / 7,466 images, frontal-view selection,
weak core labels, strict chronic-only downgrade, quality exclusions and resulting
2,371 studies. Know the train/validation/test counts and why UID grouping and
exact-image hashes do not prove patient independence. Findings/impression are
excluded from predictors; impression participates in the cohort-label rule only.

## Zeyad and Nadeen - supplied model notebooks and fusion

Study `team_classification.py`, `team_features.py` and the saved comparison CSV.
Trace both CNNs, frozen BatchNorm, staged AdamW training, AMP and early stopping.
Explain the word/character TF-IDF plus metadata text branch, 70/30 late fusion,
validation balanced-accuracy threshold 0.49, and study-mean evaluation. The local
model was retrained because classification weights were not supplied. Accuracy
is 82.87%, F1 0.7918, ROC-AUC 0.8816. ResNet50 has higher F1. Current values must
not be mixed with the older v1 cohort or the team's 80.6% notebook result.

## Mohamed Ehab - integration, evaluation, deployment and reporting

Trace `team_inference.py`, `app/main.py`, `ui/app.py`, `explainability.py` and
`report_generator.py`. Explain checksums, shared preprocessing, readiness,
locking, HTTP calls, Docker CPU inference, Grad-CAM at the ResNet50 branch and
raw-token occlusion. Know the deterministic five-section report and why the
learned generator remains experimental: documented OpenI pretraining overlap,
unproven per-test-image membership and poor finding-level checks. Do not claim
SHAP or proven image-level leakage. Use the current runbook and recorded fallback.

## Shared rehearsal

- State evaluation unit, dataset limitations and uncalibrated score meaning.
- Explain precision 83.45%, recall 75.32% and the false-positive demo.
- Explain why historical Optuna/v1 metrics are separate references.
- Identify your supplied work and AI-assisted implementation honestly.
- Rehearse the exact input text, model version, report download and fallback.

Both Docker services and the real browser workflow have been verified. Technical
checks do not establish the team's understanding; rehearse handoffs together.
