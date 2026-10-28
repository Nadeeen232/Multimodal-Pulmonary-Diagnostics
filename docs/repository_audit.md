# Repository and source audit

**Historical v1 audit.** Current Team29 notebook analysis is in
`technical_evidence_report.md`; current submission readiness and hashes are in
`final_submission_checklist.md` and `submission_manifest.json`.


Audit date: 26 October 2026, Africa/Cairo.

## Inputs reviewed

| Source | SHA-256 |
| --- | --- |
| `SIC8 - AI Project Final .pdf` (external reference) | `beebcc2f19032058668d104902c1683f318fb260b17971e19722e359f71eb093` |
| `Indiana_Chest_Xray_V2_4258.ipynb` (external reference) | `6b2dbfeb5c2b40d896ac8d1025a60cf95a4b031b15bcd388d2afbcdc860a7403` |

The PDF has 19 physical pages. Core framework: physical pages 4–6; healthcare brief and acceptance: 9–10; submission requirements: 17. The documents describe project requirements and prior implementation. They do not provide authority to override the user's file-based implementation request.

The workspace was initially empty. No exported split CSVs, TF-IDF artifacts, or CNN checkpoints were found in the project or the targeted Downloads search. The user subsequently confirmed these exports are unavailable and explicitly requested acquisition of the public Indiana dataset.

## Reusable notebook work

- Schema and duplicate checks; report-to-projection many-to-one join.
- Separation of `problems`/`mesh` label sources from predictor text.
- Grouping images by report UID and fitting TF-IDF only on the training partition.
- Conservative augmentation, balanced image loss, pretrained encoders, staged fine-tuning and early stopping.
- Explicit treatment of uncertain weak labels.

## Corrections and limits

1. Report UID is not an established patient identifier. Grouped evaluation therefore does not prove independence between patients.
2. The labeling cell borrows names from CheXpert but is a custom keyword mapping. It includes nonpulmonary cardiac and fracture findings; incidental-only cases are excluded. Manual label review and precise task naming are still necessary.
3. Ordinary StratifiedKFold on per-image rows would leak repeated report text. New tuning uses unique studies and explicit grouped folds. TF-IDF is refit within every CV training fold.
4. `stop_words="english"` removes clinical negation. New TF-IDF preserves negation.
5. Removing disease words from genuine pre-examination history is not inherently a leakage correction. New default retains indication words, excludes findings/impression/annotation fields, and excludes comparison by default because its availability is not established.
6. `ImageOps.pad` crops to fill the requested dimensions. Shared preprocessing now uses contain + borders so the entire image remains visible. This is a deliberate change: notebook weights must not be silently reused under changed preprocessing.
7. The notebook source references `os.makedirs` without an `os` import and uses `THRESHOLDS` in its last cell without defining it. Saved outputs therefore do not establish a clean, top-to-bottom reproducible run.
8. Notebook checkpoints contain only a state dictionary, without a full preprocessing/label/split manifest. They were saved to temporary Colab paths and are unavailable locally.
9. Notebook test results were already inspected. We cannot describe that historical test set as never seen. The new pipeline has documented deterministic splits and a locked final evaluation; this does not erase earlier knowledge of the dataset.

## Historical notebook outputs

| Model | Test image rows | Accuracy | ROC-AUC | Abnormal precision | Abnormal recall | Abnormal F1 |
| --- | --- | --- | --- | --- | --- | --- |
| ResNet50 | 740 | 0.793 | 0.861 | 0.82 | 0.72 | 0.77 |
| DenseNet121 | 740 | 0.792 | 0.874 | 0.83 | 0.71 | 0.76 |

These are stored notebook outputs at threshold 0.5, not reproduced results from this repository. They use different splits/preprocessing and image-level weighting, so they must not be inserted into the new ablation table as comparable experiments.

## Environment

Python 3.12 is available alongside a default Python 3.14 installation. The project uses a local `.venv` with Python 3.12. The machine reports a GTX 1660-series GPU with 6 GB VRAM. Docker was not available at audit time. The real dataset metadata downloaded successfully; full-image acquisition and CUDA setup are recorded in the local logs.
