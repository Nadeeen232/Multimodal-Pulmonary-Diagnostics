# Project log: report notebook, deployment, model retraining and the new UI

This file explains what changed in the repository after the original SIC submission, why it
changed, and how you can repeat each step yourself. It covers four pieces of work done in
October 2026 by Nadeen's account (Nadeeen232).

The live app is at <https://multimodal-pulmonary-diagnostics.streamlit.app/> and the code it runs
is in <https://github.com/Nadeeen232/Multimodal-Pulmonary-Diagnostics>.

## The dataset

Everything here uses the Indiana University chest X-ray collection (also called OpenI). We use
the Kaggle copy published by raddar:

<https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university>

The download is about 14 GB. It contains two CSV files and 7,470 PNG images:

- `indiana_reports.csv` has one row per study: the clinical indication, comparison, findings,
  impression and the structured problem tags.
- `indiana_projections.csv` maps each image file to its study and view (frontal or lateral).

You do not need a Kaggle account for it; KaggleHub downloaded it without a login. The data is
not stored in Git. It lands in `data/.cache/` inside the repository, which `.gitignore` excludes.

## 1. Final report-generation notebook

Nadeen's final report-generation experiment lives in
[`notebooks/Team29_Report_Generation_Final.ipynb`](../notebooks/Team29_Report_Generation_Final.ipynb).
It drafts a radiology report with FLAN-T5-small. The image tags now come from a classifier trained
on IU training studies only, using CheXpert-only XRV features. The earlier version used
`densenet121-res224-all` weights, which were trained partly on OpenI itself, so its results could
not be trusted.

The notebook was added unchanged. The Python loader for it,
[`src/pulmonary/learned_reports.py`](../src/pulmonary/learned_reports.py), was rewritten to read
the new export, and it now refuses exports that use the OpenI-trained weights. We checked it
against the real exported model:

- The repository loader and the notebook's own `inference_v2.py` produced identical prompts,
  probabilities and reports.
- The forced-finding step matched the notebook's saved reports on all 359 validation studies,
  with 0 differences.

Section 5 of the technical evidence report (Markdown and Word) was updated to describe this
version. The learned report is still experimental. The app shows the rule-based report.

## 2. Free deployment on Streamlit Community Cloud

Streamlit Community Cloud is free and needs no credit card, but it runs a single process. The
project normally runs as two services (FastAPI and Streamlit). To fit, the Streamlit app now
starts the FastAPI app inside its own process when `PULMONARY_IN_PROCESS=1` is set. Every request
still goes through the same routes and input checks as the real API. Docker and local use work
exactly as before.

The trained model is too large for Git, so the app downloads it on first start from a GitHub
release. It only accepts a file whose SHA-256 matches the value pinned in
[`deploy/streamlit/streamlit_app.py`](../deploy/streamlit/streamlit_app.py). If the download fails,
the app says "The model is not ready" and does not crash.

Streamlit needs admin rights on the repository it deploys. Nadeen had push access to the original
repository (mohamed1232005) but not admin, so the full history was copied to
`Nadeeen232/Multimodal-Pulmonary-Diagnostics` and the app deploys from there.

To deploy it again: create an app at <https://share.streamlit.io> from that repository, branch
`main`, main file `deploy/streamlit/streamlit_app.py`, and pick **Python 3.12** under Advanced
settings. Other Python versions fail because the pinned CPU PyTorch files are built for 3.12.

## 3. Retraining the classification model

The app needs the trained classifier bundle (ResNet50, DenseNet121 and a text model). The original
bundle, `0.2.0-team29-47e281fc`, was only on Mohamed's computer, and we could not get it in time.
The team's classification notebook did not help either: it saved its weights to Colab's temporary
disk, which is wiped when the session ends.

So we retrained it with
[`notebooks/Team29_Classification_Training.ipynb`](../notebooks/Team29_Classification_Training.ipynb).
That notebook calls the same functions as `scripts/run_team_classification.py`, with the same
settings: batch size 8, mixed precision, at most 18 epochs per CNN and early stopping. The saved
notebook contains the real outputs of this run, including the loss curves.

The run used an NVIDIA RTX 3060 Laptop GPU (6 GB), PyTorch 2.13.0 with CUDA 13.0 and
scikit-learn 1.9.1. The whole notebook took 15.5 minutes: about 4 minutes to prepare 4,543
images, 5 minutes for ResNet50 and 5.5 minutes for DenseNet121. The download took longer than the
training. The connection dropped once at 3 GB, and KaggleHub resumed from that point.

### The split is identical to the original

Data preparation writes `cohort_manifest.json`, which records the study and image lists. Its
SHA-256 starts with `47e281fc`, the same as the original. That means the retrained model saw the
same 1,660 training studies and was tested on the same 356 test studies, so the numbers below
compare like with like.

### Test results (356 studies)

| Model | Accuracy, retrained | Accuracy, original | ROC-AUC, retrained | ROC-AUC, original |
| --- | --- | --- | --- | --- |
| Text only | 0.663 | 0.663 | 0.720 | 0.720 |
| ResNet50 | 0.775 | 0.823 | 0.855 | 0.878 |
| DenseNet121 | 0.775 | 0.792 | 0.864 | 0.864 |
| Image ensemble | 0.792 | 0.817 | 0.867 | 0.876 |
| **Fusion (served)** | **0.809** | **0.829** | **0.874** | **0.882** |

The text model came out identical because its training has no randomness left once the seed is
fixed. The image models differ because GPU training is not exactly repeatable. ResNet50 lost the
most. The served fusion model is 2 points lower in accuracy and 0.008 lower in ROC-AUC. With 356
test studies, one or two points is within what a different training run can produce. We did not
retrain until the numbers matched, because that would mean tuning on the test set. The pipeline
locks the test after one evaluation for that reason.

The fusion settings were chosen on validation: image weight 0.75 and decision threshold 0.35.

The three demo X-rays get the same predicted class as before:

| Demo case | Original score | Retrained score | Class |
| --- | --- | --- | --- |
| Normal (UID 3212) | 0.213 | 0.160 | Normal |
| Abnormal (UID 2348) | 0.908 | 0.905 | Abnormal |
| Known false positive (UID 2829) | 0.711 | 0.532 | Abnormal |

### The exported bundle

The retrained bundle is called `0.2.1-team29-47e281fc`. The new version number keeps it apart
from the original, since both were trained on the same split. The export script gained a
`--model-version` option for this.

- File: `SIC_0.2.1-team29_model_bundle.zip`, 114.2 MB
- SHA-256: `b6a0fda4d3359afe0b0b5c95dde0532fcc016394c622a2f9f17088127b9c3f95`
- Published as the `model-bundle-0.2.1-team29` release on Nadeen's repository

Before publishing, we loaded the ZIP through the same pinned download path the app uses, on the
CPU-only library versions the app installs. It loaded, and the demo scores matched the GPU run.

## 4. A new look for the web app

The first deployed version used Streamlit's default look: a large yellow warning banner, a sidebar
with one checkbox, raw results stacked down the page and the report in a greyed-out text box. The
redesign keeps every feature and every API call. Only the layout and styling changed.

What you see now:

- A header with the app name and a status badge showing whether the model is ready, plus the
  model version.
- A **Study** panel on the left for the X-ray, the clinical indication and an optional comparison.
- A **Result** panel on the right. It shows Normal or Abnormal as a colored pill, the score in
  large numbers, and a scale with the 35% decision threshold so you can see how far the score is
  from it. Below that you can switch between the X-ray, the Grad-CAM heatmap or both side by
  side. The indication words that pushed the score up or down are highlighted, and the
  rule-based report is laid out section by section with download buttons.
- Three buttons that load held-out test studies from `demo_images/team29/`: a normal case, an
  abnormal case and a known miss (a normal study the model flags). Anyone can try the app
  without having an X-ray at hand. The model never trained on these three studies.
- Light and dark themes that follow the viewer's system setting. Dark mode suits X-rays, since
  radiology viewers are dark. A small script in `ui/theme.py` reads which theme Streamlit is
  showing and switches the custom colors, because Streamlit does not rerun the app when the
  theme changes.
- Clear states for "no study yet", "analyzing" (grey placeholder blocks), errors and "model not
  ready". The upload box now states the real 10 MB limit.

The colors, fonts (Geist and Geist Mono) and corner radius live in `.streamlit/config.toml`.
The custom pieces (verdict card, threshold scale, highlighted words, report layout) are built in
`ui/theme.py`. The Docker UI image copies both files and the three sample studies.

One bug came up while testing locally: adding the `ui/` folder to Python's import path made
`ui/app.py` hide the project's `app` package, which the in-process model service needs. The
theme file is now loaded straight from its path, and a test checks that `ui/` stays off the
import path.

We checked the redesign with the real model in both themes and at phone width. The three sample
cases gave the same scores as before: 16.0% (Normal), 90.5% (Abnormal) and 53.2% (Abnormal).

## Retraining it yourself

1. Install the project with a CUDA build of PyTorch 2.13.0 (see the README), plus `ipykernel`.
2. Open `notebooks/Team29_Classification_Training.ipynb` and run all cells. It downloads the
   dataset, prepares it, trains, tests once and exports to `artifacts/team_bundle/` and `dist/`.
3. To use the new bundle in the app, attach the ZIP to a GitHub release and put its URL and
   SHA-256 in `deploy/streamlit/streamlit_app.py` (or set `MODEL_BUNDLE_URL` and
   `MODEL_BUNDLE_SHA256` as Streamlit secrets).

Each step skips work that is already saved, and an interrupted CNN resumes from its best
checkpoint. The test evaluation locks after it runs. Delete `artifacts/team_classification/` to
start a fresh experiment.

## What is still the old version

The slide deck, `docs/technical_evidence_report.docx` (apart from section 5) and
`docs/submission_manifest.json` still describe the original `0.2.0` model and its 82.87%
accuracy. They are submission records, so we left them as they were.

The model learns from weak labels derived from the structured tags, not from a radiologist's
diagnosis. The test split had also been looked at during earlier development. Treat these
numbers as a research prototype result. A clinician has to check every output.
