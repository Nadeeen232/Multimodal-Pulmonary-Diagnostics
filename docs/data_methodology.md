# Data, task, and validation methodology

This page documents the **v1 reference protocol**. The separate Team29
reproduction uses a different curated target, 384-pixel preprocessing and
indication/comparison features. Its cohort and evaluation rules are documented
in [technical_evidence_report.md](technical_evidence_report.md); the comparison of data profiles
is in [data_assets.md](data_assets.md). Do not combine the two experiments' metrics.

## Provenance and pairing

Dataset: [Indiana University Chest X-rays, raddar mirror on Kaggle](https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university). Record the downloaded dataset version/path in `data/download_manifest.json` and the CSV SHA-256 hashes in the preparation manifest. The originating collection is described in [Preparing a collection of radiology examinations for distribution and retrieval](https://pubmed.ncbi.nlm.nih.gov/26133894/).

The actual CSVs contain 3,851 reports and 7,466 projection records. Each report UID may have multiple views. Pairing uses the report UID with a validated many-to-one join; image filenames must resolve uniquely. Raw images remain unchanged. Processed lossless PNGs use the identical deterministic transform used by inference.

## Data dictionary

| Field | Role |
| --- | --- |
| `uid` | Report/study identifier; not asserted to be a patient identifier |
| `filename`, `projection` | Linked image identifier and view metadata |
| `indication` | Default text predictor, a pre-examination clinical indication |
| `comparison` | Optional prior-study context; excluded by default |
| `findings`, `impression` | Current radiology interpretation; excluded from predictors |
| `problems`, `mesh` | Structured sources used only to derive weak labels |
| `medical_status`, `target` | Custom annotation proxy; normal=0, abnormal=1; uncertain excluded |
| `group_id` | Patient ID if reliably available, otherwise UID; exact duplicate image components are linked |
| `sample_id` | Stable UID + filename key for row/embedding alignment |
| `image_path` | Source image path relative to configured dataset root |
| `processed_image_path` | Cached deterministic image path relative to artifact root |
| `image_sha256` | Source image checksum for exact duplicate/linkage checks |
| `model_text` | Only permitted, cleaned predictor text |

Indiana indications are short and are not equivalent to full physician admission notes. Missing indications and placeholder-only text are audited. Training can include missing text as an empty TF-IDF vector; serving requires a nonempty indication. Missing-text performance should be reported as a limitation/subgroup before making claims about robustness.

## Weak labels

`indiana-weak-v1` preserves the team's keyword mapping for an auditable starting point. An exact `problems=normal` is negative unless a core abnormal term conflicts, in which case the label is uncertain. A core keyword elsewhere in structured annotations is positive. Everything else is uncertain, including incidental-only annotations. Matching respects word boundaries.

This is not the official CheXpert labeler. It has not been clinically adjudicated. The mapping includes cardiomegaly and fractures as well as pulmonary/pleural terms, so the implemented target is a broad annotation-based thoracic abnormality proxy. A pulmonary-only interpretation is not justified. Before a final task claim, review positive/negative/uncertain samples, revise the label specification if required, version it, and rerun all affected models. Never revise labels to improve observed test scores.

Actual inherited report labels: 1,379 normal, 1,222 abnormal, 1,250 uncertain. Excluding uncertain reports changes the evaluated population and may inflate apparent performance. The system cannot diagnose any of the individual keyword conditions.

## Leakage controls

1. Split whole report/patient groups, with approximate group-level class stratification (70/15/15). Do not call UID a patient identifier.
2. Link groups that share identical image hashes before splitting; reject residual overlap.
3. Fit TF-IDF and feature scalers on train only. Refit TF-IDF inside grouped CV folds.
4. Train text models on unique UIDs; probability calibration folds are explicit grouped indices.
5. Exclude current findings, impression, problems, and MeSH from predictive features.
6. Use validation only for model selection, early stopping, fusion weight, and thresholds. Freeze the selected deployment bundle before final test evaluation.
7. Verify split-file hashes when loading them and registry-to-split hashes before fusion/export.

Known disease words in a genuine pre-examination indication are retained. Their presence is not automatically leakage, but the timing and origin of that field remain a dataset limitation. Repeated use of validation for several choices can overfit it. Historical notebook test results have already been observed; external validation is absent.

## Metrics and review cost

Primary metrics are study-level: average image probabilities within each UID and compare once to the study label. Report accuracy, abnormal precision/recall/F1, ROC-AUC, average precision (the reported PR-AUC convention), Brier score, confusion matrix, support, and chosen threshold. Single-image API requests do not aggregate multiple views; study-level experiments and single-view serving should not be conflated.

The example FN:FP cost ratio 10:1 is a sensitivity scenario, not established clinical economics. Always compare with reviewing every study. Weak text scores may minimize this assumed cost by flagging nearly everything, which provides little queue reduction. No real clinical benefit or workload reduction may be claimed without supporting evidence.
