"""Rebuild the current submission's Markdown evidence and runbook from saved results."""

import json
from pathlib import Path

import pandas as pd
from build_tracking_evidence import tracking_markdown

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "reports/evidence/team29"
MODEL_NAMES = {
    "text_only": "Text-only — Logistic Regression",
    "resnet50": "ResNet50",
    "densenet121": "DenseNet121",
    "image_ensemble": "Image ensemble",
    "multimodal_fusion": "Multimodal Fusion",
}


def main():
    table = pd.read_csv(EVIDENCE / "ablation_results.csv")
    model = json.loads((EVIDENCE / "model_info.json").read_text())
    metrics = json.loads((EVIDENCE / "multimodal_fusion_metrics.json").read_text())
    deployment = json.loads((EVIDENCE / "deployment_verification.json").read_text())
    rows = "\n".join(
        f"| {MODEL_NAMES[r.model]} | {100 * r.accuracy:.2f}% | {r.f1:.4f} | {r.roc_auc:.4f} | {r.threshold:.2f} |"
        for r in table.itertuples()
    )
    confusion = []
    for row in table.itertuples():
        m = json.loads((EVIDENCE / f"{row.model}_metrics.json").read_text())
        (tn, fp), (fn, tp) = m["confusion_matrix"]
        confusion.append(f"| {MODEL_NAMES[row.model]} | {tn} | {fp} | {fn} | {tp} |")
    report = f"""# Technical Evidence & Working Solution

Multimodal Clinical Decision Support for Pulmonary Pathology — SIC AI Track

Current submission: 28 October 2026. All primary results below belong to
**{model["model_version"]}**. This replaces the previous technical evidence
document's current-model narrative; historical v1 records remain separately available.
Source: SIC handbook, page 17, “The complete submission package”.

## 1. System, Data Assets and Reproduction

| Property | Current value |
| --- | --- |
| Model bundle | `artifacts/team_bundle/manifest.json` |
| Classifier | ResNet50 + DenseNet121 image ensemble, TF-IDF Logistic Regression |
| Fusion | 70% image ensemble + 30% text |
| Decision threshold | 0.49; selected by validation balanced accuracy |
| Evaluation unit | Study mean across available views; UI input is one image |
| Default report | Rule-based evidence summary; clinician verification required |
| Source modules | `team_classification.py`, `team_features.py`, `team_inference.py` |

![Current training and serving architecture](architecture.png)

The Indiana University collection is distributed through NLM Open-i; this project
uses the `raddar/chest-xrays-indiana-university` Kaggle mirror. The local source has
3,851 reports and 7,466 image records. The selected cohort requires a frontal view,
retains at most one frontal and one lateral image and excludes uncertain and
poor-quality cases. It contains 2,371 studies and 4,543 image rows.

| Split | Studies | Image rows |
| --- | ---: | ---: |
| Train | 1,660 | 3,179 |
| Validation | 355 | 683 |
| Held-out test | 356 | 681 |

| Field | Definition / permitted use |
| --- | --- |
| uid | Report/study identifier; not a verified patient identifier |
| filename / projection | Image pairing and frontal/lateral view metadata |
| indication | Pre-examination clinical context; classifier input |
| comparison | Prior-study context; optional classifier input |
| findings / impression | Current interpretation; excluded from classifier predictors |
| Problems / MeSH | Structured weak-label sources; never classifier features |
| target | Normal=0, Abnormal=1; uncertain cases excluded |

Normal requires the exclusive Normal problem tag with no core abnormal tag.
Abnormal uses ten core groups, including cardiomegaly and fractures as well as
pulmonary findings. Contradictory cases are uncertain. A strict labeling rule moves
57 chronic-only cases with a first-sentence “no acute” impression to uncertain.
That impression is used for cohort labeling, not as a predictor. The target is a
broad weak thoracic proxy, not an adjudicated pathology diagnosis.

Both views of a UID stay together. The selected cohort's SHA-256 audit found no
exact duplicated image rows and required no split reassignment. Near duplicates
and true patient independence are not established. Shared preprocessing uses
grayscale, aspect-preserving Lanczos padding to 384 × 384, RGB conversion and
ImageNet normalization. Data dictionary and both experiment profiles are in
`docs/data_assets.md`; source hashes are in the cohort manifest.

## 2. Experiment Tracking & Tuning

| Training component | Recorded configuration |
| --- | --- |
| Image training | 3 head + up to 15 fine-tuning epochs; batch size 8 |
| Optimization | AdamW; ReduceLROnPlateau; class-weighted cross-entropy |
| Early stopping | Patience 2 on validation loss; frozen BatchNorm statistics |
| Retained checkpoints | ResNet50 epoch 7; DenseNet121 epoch 10 |
| Completed image epochs | ResNet50 10; DenseNet121 12 |
| Text search | 6 C values × 2 class-weight settings = 12 validation comparisons |
| Fusion search | 21 image weights; validation ROC-AUC objective |
| Threshold search | Validation balanced accuracy; test not used for selection |
| Tracking | Separate `artifacts/team_classification/mlflow.db` |

TF-IDF word/character vocabularies and metadata scaling are fitted on training
data only. Indication/comparison are cleaned and label terms masked. Final
selection is frozen in `selection.json` before the test evaluation lock is written.
Image training and final evaluations are tracked with manifests, hashes, metrics
and plots. The test results were not used to select the deployed weight or cutoff.

![ResNet50 training and validation loss](../reports/evidence/team29/resnet50_loss.png)

![DenseNet121 training and validation loss](../reports/evidence/team29/densenet121_loss.png)

The GTX 1660 SUPER run uses AMP with a tested legacy-cuDNN benchmark configuration.
Deterministic cuDNN FP16 returned NaNs; invalid forward passes were recomputed in
FP32. Backend correction and a disk-exhaustion interruption required checkpoint
restarts and optimizer reinitialization. Interrupted MLflow runs retain their
failure/cancellation statuses. Seeds are recorded, but bitwise reproducibility is
not promised with benchmark-selected GPU algorithms. This is a local reproduction
of the team's approach, not their unavailable classification weights.

Historical v1 Optuna tuning completed 20 trials and is retained as development
evidence. It belongs to a different cohort/text pipeline and is not claimed as
new tuning for this model. Its linear/tree/CNN comparisons remain historical.

{tracking_markdown()}

## 3. Held-out Model Comparisons

All rows below use the same current test cohort, 356 studies / 681 source images.
Accuracy and F1 use each model's own validation-selected threshold. F1 is for the
Abnormal weak-label class.

| Model | Accuracy | F1 | ROC-AUC | Threshold |
| --- | ---: | ---: | ---: | ---: |
{rows}

![Current held-out ablation comparison](../reports/evidence/team29/ablation.png)

The deployed fusion achieves precision **{100 * metrics["precision"]:.2f}%**, recall
**{100 * metrics["recall"]:.2f}%** and average precision **{metrics["pr_auc_average_precision"]:.4f}**.
It slightly improves accuracy and ROC-AUC over ResNet50, while ResNet50 has the
higher F1. No statistical superiority or clinical effectiveness is established.
The old 10:1 cost policy is not used in this model. Its earlier, high-recall v1
scores used a different cohort and are not directly comparable.

| Model | True negative | False positive | False negative | True positive |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(confusion)}

![Deployed fusion ROC, precision-recall and confusion matrix](../reports/evidence/team29/multimodal_fusion_curves.png)

![Text-only ROC, precision-recall and confusion matrix](../reports/evidence/team29/text_only_curves.png)

![ResNet50 ROC, precision-recall and confusion matrix](../reports/evidence/team29/resnet50_curves.png)

![DenseNet121 ROC, precision-recall and confusion matrix](../reports/evidence/team29/densenet121_curves.png)

![Image ensemble ROC, precision-recall and confusion matrix](../reports/evidence/team29/image_ensemble_curves.png)

![Raw fusion-score reliability against weak labels](../reports/evidence/team29/multimodal_fusion_raw_reliability.png)

The reliability plot is descriptive; no probability calibrator was fitted. Scores
remain uncalibrated and must not be interpreted as disease probabilities.

## 4. Explanation Plots and Written Reports

Grad-CAM differentiates the final fused score at the ResNet50 branch. It explains
that branch's contribution, not a combined anatomical map of both CNNs. Token
occlusion removes one raw token and measures the change in the final score with
image scores fixed. This is **token occlusion, not SHAP**. Changes are not additive
causal effects. Image coordinates do not establish anatomy or patient laterality.

| Selected test example | Weak label | Model outcome | Single-image score |
| --- | --- | --- | ---: |
| UID 3212 | Normal | True negative | 21.34% |
| UID 2348 | Abnormal | True positive | 90.79% |
| UID 2829 | Normal | False positive | 71.06% |

![Normal example — actual deployed-model image and text attribution](../reports/evidence/team29/xai_uid_3212.png)

![Abnormal example — actual deployed-model image and text attribution](../reports/evidence/team29/xai_uid_2348.png)

![False-positive example — actual deployed-model image and text attribution](../reports/evidence/team29/xai_uid_2829.png)

The cases were selected after evaluation to demonstrate behavior, not to estimate
performance. Study metrics average views; these interface scores use one image.
Exact indication/comparison fields are in `demo_images/demo_cases.csv`.

The rule-based report contains Predicted Class, Impression, Visual Findings,
Clinical Context and Recommendation. It describes scores and attribution rather
than asserting that attention proves a lesion. Every report retains
“Clinician verification is required.”

![Current Streamlit written report](../reports/report_screenshot.png)

## 5. Team Notebook Analysis & Data Leakage Prevention

The team's classification notebook was inspected and ported into source modules.
Missing trained classification exports required local retraining. Image/text
alignment, permitted predictors, train-only fitting and grouped evaluation were
checked. Ambiguous image names/conflicting UIDs are rejected, exact-image hashes
are audited, and thresholds/fusion choices are selected on validation only.

The team's report generator uses FLAN-T5-small with ResNet18 visual tokens and
XRV pathology tags. Its XRV `densenet121-res224-all` configuration explicitly lists
**OpenI** among pretraining sources, and Indiana IU radiographs come from the
OpenI/NLMCXR collection. This documents **dataset-source pretraining overlap**.
Without the upstream per-image training manifest, it does **not** prove that a
particular local test image or patient was seen. Independent evaluation of this
component remains unestablished; it is not accurate to claim proven image-level
test leakage or quantify its effect from the available files.

Sources: [XRV model definition](https://github.com/mlmed/torchxrayvision/blob/master/torchxrayvision/models.py)
and [XRV OpenI dataset implementation](https://github.com/mlmed/torchxrayvision/blob/master/torchxrayvision/datasets.py).
The deployed classifier uses ImageNet weights and does not use this XRV component.

| Reporting option | Evidence | Decision |
| --- | --- | --- |
| Rule-based evidence summary | Deterministic, faithful to classifier/XAI inputs; not a learned radiology report | Default in API/UI |
| Team learned FLAN-T5 draft | Export reload reproduced its saved example; fluent text with omission risks and unresolved upstream overlap | Experimental adapter only |

The supplied learned-generator test checks found cardiomegaly in 7/23 reference
cases, atelectasis in 1/45, opacity in 9/70 and granuloma in 3/49. These are limited
lexical checks, not clinician-adjudicated scores, but demonstrate why fluency is
insufficient. Its supplied BLEU 17.30 / ROUGE-L 0.377 / METEOR 0.326 are language
similarity metrics, not diagnostic accuracy. Image swaps moved both visual tokens
and XRV tags, so they do not isolate the visual-token branch. Two supplied V2
folders represent different runs and were not merged into one result. The full
notebook analysis and reporting decision are consolidated in this report.

## 6. Functional Tests and Working Solution

| Check | Saved evidence | Result |
| --- | --- | --- |
| Focused regression tests | `reports/evidence/team29/pytest.txt` | 13 passed |
| Export/software parity | Team bundle checksums, direct-vs-fused score tests | Passed |
| API prediction/explanation | `deployment_verification.json` | HTTP 200; matching version/scores |
| Invalid missing image | Same deployment verification | HTTP 422 |
| Three real demo predictions | `demo_prediction_verification.json` | All matched local export |
| Container UI report rendering | Same deployment verification | Passed |
| Real browser upload/report/download | `automated_demo.json` | Passed |
| Both Compose services | Saved Docker evidence and live `/ready` | Healthy |

The 13 tests cover predictor exclusion, aspect-preserving padding, study-label
consistency, validation thresholds, import checksum rejection, real fusion parity,
Grad-CAM/report generation, API validation/readiness and Streamlit rendering.
Artificial test weights are software fixtures and are not the source of quality
metrics. Real model quality comes from the frozen held-out outputs above.

```text
............. [100%]
13 passed in 24.75s
```

| Runtime component | Configuration |
| --- | --- |
| API | FastAPI, CPU inference, localhost:8080 |
| UI | Streamlit, localhost:8501; internal API URL http://api:8080 |
| Model mount | `artifacts/team_bundle` → `/models`, read-only |
| Model version | {model["model_version"]} |
| Recorded /predict latency | {deployment["checks"]["predict"]["latency_seconds"]:.2f} seconds for selected case; not a benchmark |
| Failure behavior | Unavailable model → readiness/inference 503; no invented scores |

![Actual current FastAPI Swagger UI](../reports/swagger_screenshot.png)

![Docker Desktop container list — header omitted for privacy](../reports/docker_desktop_screenshot.png)

![Actual current Streamlit interface](../reports/ui_screenshot.png)

Start from the project root:

```powershell
docker compose up -d --no-build --wait --wait-timeout 180
docker compose ps
Invoke-RestMethod http://localhost:8080/ready
Invoke-RestMethod http://localhost:8080/model-info
```

A fresh checkout needs a trusted complete model bundle first. Model binaries and
raw data are excluded from Git. The offline bundle archive is packaged separately
under `dist/`; restore its files into `artifacts/team_bundle/`. A clean dependency
build uses `docker compose build`; an existing trusted image supports the smaller
code-only `Dockerfile.update` build. See README and `docs/demo_runbook.md`.

## 7. Submission Package and Limits

| SIC item | Artifact / evidence |
| --- | --- |
| 01 Source code | `src/`, `app/`, `ui/`, `scripts/`, `tests/` |
| 02 Technical documentation | README, architecture, dependencies and execution guides |
| 03 Data assets | `docs/data_assets.md`, provenance, dictionary, split manifest |
| 04 Technical evidence | This report, tracked searches, comparison/XAI plots and tests |
| 05 Working solution | Versioned model bundle, offline archive and verified Compose stack |
| 06 Presentation | `docs/SIC_Final_Presentation.pptx`, current notes and inserted assets |
| 07 Demonstration | `docs/demo_runbook.md`, automation and recorded fallback |
| 08 Team contribution summary | `team_contributions.md` |

The exact checklist is recorded in `docs/final_submission_checklist.md`. Technical
package checks do not substitute for each member's rehearsal or understanding.
There is no external clinical validation, verified patient grouping or prospective
workflow-benefit measurement. Historical test results were already visible during
project development; this is not a new independent external validation study.
Clinician verification is required.
"""
    (ROOT / "docs/technical_evidence_report.md").write_text(report, encoding="utf-8")
    print("Updated technical evidence Markdown from current saved results")


if __name__ == "__main__":
    main()
