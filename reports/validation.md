# Current validated release - 28 October 2026

Model **0.2.0-team29-47e281fc** is deployed in healthy API/UI containers.
The current focused regression suite passed **13 tests in 24.75 seconds**.
Real predictions, explanations, three demo cases, browser upload/report/download
and a recorded fallback were verified. Held-out accuracy is **82.87%** and ROC-AUC
**0.8816** on 356 studies. Current evidence is `reports/evidence/team29/` and
`docs/technical_evidence_report.docx`; see the eight-item submission checklist.

## Historical validation chronology

The entries below describe earlier v1/artificial development stages. Statements
that Docker was unavailable or training was ongoing are historical, not current.

# Validation record

26 October 2026. Initial execution: Python 3.12 local virtual environment, CPU Torch 2.14.1 and torchvision 0.29.1.

Initial focused suite: **9 passed** in 21.36 seconds. It checked weak-label conflicts, negation preservation, no target-field input, image aspect-preserving borders, study-level aggregation, threshold selection, priority bounds, API validation, unavailable model behavior, and real early-fusion Torch prediction/Grad-CAM/text occlusion/checksum rejection. The real Torch fixture uses explicitly random weights in a temporary test directory; it does not establish trained-model quality.

The updated suite passed **12 tests in 16.23 seconds**, including full preparation with artificial images, exact-duplicate group linkage, split-file tampering detection, and Streamlit startup/required-input behavior. This run used CUDA Torch 2.13.0 and torchvision 0.28.0. One dependency deprecation warning concerned Starlette's httpx TestClient adapter; it did not fail the suite. An initial UI test had an incorrect relative file path; the test was corrected to use the absolute project path and passed.

The complete software pipeline also passed on a separate 40-study/80-image artificial fixture: preparation, two Optuna trials, all four text candidates (including dummy), both CNNs, both fusion strategies, validation-based export, final evaluation, explanation examples, report generation, and the real `/explain` API route. Official pretrained ImageNet weights were downloaded; fixture training used CUDA. Logs: `reports/software_smoke_final.log`. Fixture path: `.tmp/software-smoke-15e030ce`. This is **software evidence only**, not a clinical or real-dataset model comparison.

`ruff check` and `pip check` passed. Core CPU serving constraints match the locally verified library versions and the CUDA trainer's Torch release. Docker itself is still unverified.

Final evidence-stage rerun: **13 tests passed in 17.90 seconds**. The added regression test verifies that weighted validation loss matches whole-dataset weighted cross entropy, independent of batch composition. Each CNN now receives a freshly seeded loader. The full artificial workflow passed again with dataset audit plots, label-review samples, category-aware XAI selection, combined case cards, priority distribution and report generation. Latest fixture: `.tmp/software-smoke-173ddabb`; log: `reports/software_smoke_evidence.log`. PowerPoint generation also succeeded against this explicitly artificial fixture; no final clinical-results deck has been claimed.

## User-requested GPU limits and mixed precision

The user confirmed a GTX 1660 SUPER with 6 GB VRAM and requested batch size 8, AMP and a 15–20 epoch maximum. Training now rejects a batch size other than 8, budgets at most 20 total epochs, defaults to 18 CNN epochs (3 head + 15 fine-tune) and at most 20 fusion epochs, and retains early stopping with patience 3. CUDA autocast and GradScaler are enabled; CPU execution disables AMP. Validation and fusion batches are also limited to 8.

Actual short GPU check at 224×224 with batch 8: ResNet50 peak allocated 456.6 MiB / reserved 472 MiB; DenseNet121 peak allocated 216.4 MiB / reserved 252 MiB. Both produced FP16 classifier outputs and updated their heads with GradScaler. These are observed short-check values, not a guarantee for an entire training run or competing GPU applications. Full details: `reports/generated/gpu_budget_check.json`.

The first 64×64 artificial AMP workflow exposed a nonfinite ResNet convolution output. The shared loop now logs the event and recomputes an affected batch in FP32 while keeping AMP enabled for normal batches. A nonfinite FP32 result still stops the run. Cumulative fallback counts are logged to MLflow. Regression coverage verifies FP32 recovery, failure on nonfinite FP32, and all enforced configuration limits.

Latest suite: **21 tests passed in 26.71 seconds**, with the same dependency deprecation warning. The complete artificial workflow then passed with AMP and the logged recovery path: `.tmp/software-smoke-825bec30`, log `reports/software_smoke_amp_stable.log`. Full real-data preprocessing/training is a separate ongoing workflow.

## Port 8080 and completed real-model local stack

The full real-data workflow completed successfully. The initial local stack verification stopped because port 8000 was occupied. At the user's request, Docker ports/healthcheck/startup command, Compose mappings, Streamlit backend defaults, local verification/finalization scripts and README commands were changed to API port 8080. Streamlit remains on 8501. Port checks found both target ports free before startup.

The finalization workflow then passed, including real-results PowerPoint generation and local service verification using public validation study UID 1001 and frozen model `0.1.0-fusion_early-e12abca4`. `/ready` and `/docs` returned 200; `/predict` returned 200 in 0.132 seconds; `/explain` returned 200 in 0.256 seconds; missing-image input returned 422; Streamlit health returned 200. A separate Streamlit AppTest rendered the actual UI against the live API and confirmed its model-ready state without exceptions. These timings are single-request observations, not a load benchmark.

The focused API/UI regression rerun passed **4 tests in 9.80 seconds**, with the known dependency deprecation warning. Full evidence and service process IDs are in `reports/generated/functional/local_stack_verification.json`. The original held-out metrics and model selection were preserved. Docker execution remains deferred; its configuration changes have not been container-tested.

Real metadata-only training completed 20 grouped Optuna trials and fit logistic regression, calibrated linear SVM, and ExtraTrees. The original public CSV schema and inherited label counts were confirmed. No image/fusion model has been trained in this workspace yet. The source notebook's saved CNN outputs are historical evidence only.

At that earlier check Docker was not installed, so container acceptance had not been executed.

## Docker verification — 26 October 2026

With Docker Desktop and WSL enabled, `docker compose build` and
`docker compose up -d --wait --wait-timeout 180` completed successfully.
Both the API (8080) and Streamlit (8501) containers reported healthy.
The API uses CPU Torch and the read-only frozen model bundle.
Readiness, Swagger docs, UI health and UI page returned 200. Demo UID 1330
returned the expected prediction in 0.206 seconds; explanation returned a
valid PNG Grad-CAM in 0.287 seconds. Missing-image input returned 422.
Streamlit AppTest executed inside the UI container, rendered without
exceptions, and displayed Model ready using the internal Docker API URL.
These timings describe individual requests, not a load benchmark.
Evidence: `reports/generated/functional/docker_stack_verification.json`.
The repeatable checks are in `scripts/verify_docker_stack.py`; PowerShell demo
instructions are in `docs/docker_demo.md`. Containers were left running.

## Rule-based report feature — 26 October 2026

Added `src/report_generator.py`, measured Grad-CAM region summaries, signed text
attribution descriptions and report fields on both inference endpoints. The UI
renders a read-only report and offers text download. Existing model weights,
selection and operating threshold were preserved. The API still requires an
X-ray and indication; this feature does not implement learned image-to-text.

Seven focused report/API/real-model/UI tests passed in 15.46 seconds, with the
known Starlette TestClient deprecation warning. Lint passed. An initial sandboxed
test attempt could not access temporary directories; the rerun with access passed.
Sample reports were generated for all three selected demo cases.

The regular Docker rebuild was interrupted during very slow dependency downloads.
Both images then rebuilt successfully with the optional trusted local dependency
base (`docker compose build --build-arg DEPENDENCY_BASE=sic_graduation_project-api:latest`).
The Dockerfile retains its clean Python base by default. Compose replaced both
containers, and both reported healthy. Updated container verification passed:
report sections and readable region on both endpoints, prediction consistency,
valid Grad-CAM PNG, invalid-input rejection and actual report rendering inside
the Streamlit container. Evidence was refreshed in
`reports/generated/functional/docker_stack_verification.json`.

## Clinical report format refinement — 26 October 2026

Reports and Streamlit now omit educational-use wording and use five sections:
Predicted Class, Impression, Visual Findings, Clinical Context and Recommendation.
Both inference routes expose a human-readable `predicted_class` alongside the
existing machine label. The class follows the original operating threshold.
Signed token associations and readable image-space attention descriptions are
retained without claiming anatomical findings, disease probability or lesion absence.
Reports retain "Clinician verification is required".

Seven focused tests passed in 16.13 seconds, plus the added UI wording assertion
passed in 3.77 seconds. Lint passed. Both Docker images rebuilt using the trusted
local dependency cache, both replacement containers became healthy, and the
updated container checks passed: five report sections on both inference routes,
expected Normal/Abnormal class, no educational wording, real Grad-CAM and
actual report rendering inside Streamlit. All three demo report text files were
refreshed through the Docker API. No trained model or decision threshold changed.
