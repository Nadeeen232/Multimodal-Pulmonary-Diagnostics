# Implementation handoff

26 October 2026. This file records the first implementation milestone. Runtime JSON files below are authoritative for ongoing work.

## Implemented source workflow

- Configurable data audit, conservative weak labels, report/patient grouping with exact-image duplicate linkage, split checksums, lossless shared image transform and cache.
- Training-only TF-IDF and grouped CV/Optuna; dummy, logistic, calibrated linear SVM and ExtraTrees comparisons.
- GPU-capable staged ResNet50 and DenseNet121 training with validation early stopping.
- Ordered frozen embeddings, early multimodal MLP and bounded late fusion.
- Validation selection and cost sensitivity; checksummed export before a separately locked final test evaluation.
- Grad-CAM of the actual deployed multimodal score; cached-image raw-token occlusion; category-aware validation explanation examples and case cards.
- FastAPI, validated multipart requests, readiness, model metadata and request metrics; Streamlit results and explanations with stale-result protection.
- Docker definitions, pinned core CPU serving constraints, documentation, architecture PNG/SVG and result-driven PowerPoint builder.

## Executed evidence

- Real public metadata: 3,851 reports, 7,466 projection records; no unmatched UID pairs in the CSV join.
- Real metadata-only text run: 20 Optuna trials; validation ROC-AUC 0.6548 logistic, 0.6536 SVM, 0.6362 ExtraTrees. These are provisional validation metrics; image pairing/duplicates are not verified in this run. The assumed cost rule produced near-universal review.
- 13 focused tests passed; lint and installed dependency consistency checks passed.
- Complete artificial-data software workflow and presentation export passed. This is not real-image performance evidence.
- CUDA 12.6 Torch 2.13.0 is available on NVIDIA GTX 1660 SUPER. `reports/environment.json` records the environment.

## Active background work

- Downloader: `scripts/download_dataset.py`, using the resumable KaggleHub archive under `data/.cache/`. Progress: `data/download_status.json`; log: `reports/dataset_download.log`.
- Sequential real-data worker: `scripts/run_pipeline.py --wait-for-download`. Status: `reports/generated/pipeline/status.json`; process record: `reports/pipeline_process.json`. Individual stage logs are stored in `reports/generated/pipeline/`.
- After extraction the worker creates `configs/local.yaml`, audits/preprocesses images, trains text and both CNN baselines, trains both fusions, exports the selected bundle, evaluates frozen candidates on test, saves explanations and generates the results report.
- Do not start duplicate download/training jobs. If a stage fails, the worker stops and records the failing stage; inspect its log and preserve existing final-evaluation locks.

## Remaining observed work

- Full-data image/fusion metrics and the final trained serving bundle are pending the active download/training workflow.
- After that workflow completes, run `scripts/build_presentation.py --config configs/local.yaml` and `scripts/verify_local_stack.py --config configs/local.yaml` for the actual deck and live API/UI evidence.
- Docker is unavailable on this machine, so container build/run remains unverified.
- The user supplied assigned team responsibilities and a Thursday submission deadline, with implementation targeted for today, 26 October. `team_contributions.md` and `docs/team_study_guide.md` record the roles and study plan. Actual completed contributions still require team confirmation. The label mapping remains a broad thoracic annotation proxy requiring explicit scope and label review.
