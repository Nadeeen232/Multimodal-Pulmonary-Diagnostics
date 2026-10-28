# Data Assets

This document describes the data used by Pulmonary Review Assistant: where it
comes from, how records are paired, how model inputs are prepared, and which
labels and splits define the experiment. Counts refer to the downloaded mirror
and the versioned experiments below, rather than all versions of the original collection.

## Experiment profiles

The detailed sections below describe the original **v1 reference experiment**.
The file-based team reproduction has its own label rule, split manifests and
preprocessing; these artifacts must not be mixed.

| Profile | Binary studies | Train / validation / test studies | Image input | Text input |
| --- | ---: | --- | --- | --- |
| Reference v1 | 2,601 | 1,820 / 390 / 391 | 224 × 224 padding | Indication |
| Team29 reproduction | 2,371 | 1,660 / 355 / 356 | 384 × 384 Lanczos padding | Cleaned indication + comparison |

Team29 requires a frontal view, selects up to one frontal and one lateral image,
excludes poor-quality cases and uses ten core abnormality tag groups. Its strict
rule moves chronic-only cases with a first-sentence “no acute” impression to
uncertain. Findings/impression are excluded from classifier predictors. The
impression participates in cohort labeling only. See the complete
[team model review](technical_evidence_report.md) for definitions, audits and limitations.

## Provenance and source

| Item | Description |
| --- | --- |
| Originating collection | Indiana University chest radiographs and associated reports, distributed through the National Library of Medicine's Open-i |
| Collection publication | Demner-Fushman et al., *Preparing a collection of radiology examinations for distribution and retrieval*, JAMIA; DOI: `10.1093/jamia/ocv080` |
| Project acquisition source | [Chest X-rays (Indiana University), raddar mirror on Kaggle](https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university) |
| KaggleHub identifier | `raddar/chest-xrays-indiana-university` |
| Local cached version | `versions/2` in the completed acquisition |
| Report metadata | `indiana_reports.csv`: 3,851 records |
| Image/view metadata | `indiana_projections.csv`: 7,466 records |
| Image assets consumed | Decoded PNG images; the service accepts PNG/JPEG, not raw DICOM |

The original acquisition and deidentification process is described in the
[primary collection publication](https://pmc.ncbi.nlm.nih.gov/articles/PMC5009925/).
Open-i is the originating distribution source; Kaggle is the mirror used by this
implementation. The CSV counts above are verified locally and need not equal
the original publication's collection totals.

The project does not assign a new license to third-party data. Consult the upstream
collection and mirror terms before redistributing raw or derived assets. Raw data
and model binaries are excluded from Git; this document supplies provenance and
reproduction references instead.

## Acquisition and inventory

From the repository root, with the `data` dependency extra installed:

```powershell
.\.venv\Scripts\python.exe -m pulmonary.cli download
.\.venv\Scripts\python.exe -m pulmonary.cli --config configs/local.yaml prepare
```

The download command uses KaggleHub, records the source and resolved download path
in `data/download_manifest.json`, and creates `configs/local.yaml` with the
appropriate `data_root`. The full archive is approximately 14 GB; extraction and
processed caches require additional space.

| Location | Asset / purpose |
| --- | --- |
| `data/.cache/` | KaggleHub dataset cache, including the extracted image collection |
| `data/raw/indiana/` | Initial local metadata copies; these alone are insufficient for image training |
| `data/download_manifest.json` | Acquisition source and resolved dataset path |
| `configs/local.yaml` | Local dataset location; extends `configs/default.yaml` |
| `artifacts/image_cache/` | Deterministic processed PNG cache |
| `artifacts/splits/` | Split CSVs, excluded uncertain cases and audit manifest |
| `artifacts/embeddings/` | Generated image embeddings and row-alignment identifiers |
| `reports/generated/data/` | Data overview, missingness and label-review evidence |

Preparation recursively requires exactly one copy of each metadata CSV under the
configured root. Point `data_root` at the actual dataset, not a parent directory
containing both metadata copies and the full mirror.

## Raw data dictionary

CSV identifiers are loaded as strings to preserve join keys. Column names are
trimmed and lowercased internally; for example, `MeSH` becomes `mesh` and
`Problems` becomes `problems`.

### `indiana_reports.csv` — one report per UID

| Field | Type | Meaning | Role in this project |
| --- | --- | --- | --- |
| `uid` | String identifier | Report/study key linking metadata to one or more images | Join and split grouping; not a model feature or verified patient ID |
| `indication` | Free text | Reason for the examination or associated clinical context | Default text predictor; not a complete admission note |
| `findings` | Free text | Current radiologist description of the examination | Retained for provenance; excluded from predictor inputs |
| `impression` | Free text | Current radiologist conclusion/summary | Retained for provenance; excluded from predictor inputs |
| `comparison` | Free text | Prior examination comparison information, if present | Optional eligible predictor; excluded by the current default configuration |
| `MeSH` / `mesh` | Structured text | Medical Subject Headings-style annotation tags | Weak-label derivation only |
| `Problems` / `problems` | Structured text | Report-level problem annotations, including `normal` | Weak-label derivation only |
| `image` | Text | Examination description, such as “Xray Chest PA and Lateral” | Metadata; not pixel data or the image filename |

### `indiana_projections.csv` — one row per image record

| Field | Type | Meaning | Role in this project |
| --- | --- | --- | --- |
| `uid` | String identifier | Associated report/study key | Many images may join to one report |
| `filename` | String | Image filename, including suffixes such as `.dcm.png` | Resolves an actual image file; suffix does not mean the file is DICOM |
| `projection` | Categorical text | Recorded view, such as Frontal or Lateral | Audit/context metadata; not an additional predictor |

Missing fields and deidentification markers such as `XXXX` are audited. Literal
`None.` can also appear as source text; it is not a verified clinical negative.
The API requires a nonempty usable indication, while training can represent a
missing indication with an empty TF-IDF vector. This difference limits claims
about robustness to missing text.

## Derived data dictionary

| Field / asset | Definition |
| --- | --- |
| `medical_status` | Weak label: `normal`, `abnormal` or `uncertain` |
| `target` | Binary training label: Normal = 0, Abnormal = 1; no binary target for excluded uncertain cases |
| `group_id` | Group used for partitioning; report UID in this dataset, with linked exact-duplicate image components kept together |
| `sample_id` | Stable `uid:filename` key for row/embedding alignment |
| `image_path` | Source image path relative to the configured dataset root |
| `processed_image_path` | Processed PNG path relative to the artifact root |
| `image_sha256` | Source file checksum used to identify exact duplicates |
| `model_text` | Cleaned, permitted predictor text with field prefixes |
| `manifest.json` | Input/split hashes, label/transform versions, counts and audit results |

Serving uses the machine labels `proxy_negative` and `proxy_abnormal`, plus the
human-readable `predicted_class` values Normal and Abnormal. These output labels
do not change the weak-label nature of the task.

## Pairing, cleaning and audit

1. Read metadata as strings, normalize column names and remove exact duplicate rows.
2. Reject blank UIDs, conflicting duplicate report UIDs and repeated image filenames
   that require manual pairing review.
3. Join projection rows to reports using a validated **many-to-one UID join**.
   Record unmatched projection rows and exclude them from paired data.
4. Resolve image files, verify readability, record dimensions and compute source
   SHA-256 checksums. Raw image files remain unchanged.
5. Derive versioned labels, retain the uncertain pool separately, and prepare the
   eligible binary dataset.
6. Link groups sharing identical image hashes before partitioning. Reject any
   residual group or exact-image overlap between partitions.

The completed audit found zero duplicate metadata rows, zero unmatched projection
rows and no image-read errors. Six image rows participated in exact-duplicate
linkage; this count is rows in duplicate sets, not six duplicate pairs.

Implementation: [`src/pulmonary/data.py`](../src/pulmonary/data.py).

## Image preprocessing

The shared implementation in [`src/pulmonary/images.py`](../src/pulmonary/images.py)
defines transform version `grayscale-contain-pad-bilinear-imagenet-v1`.

| Step | Operation |
| --- | --- |
| Orientation | Apply EXIF transpose where present |
| Intensity channels | Convert the decoded image to grayscale |
| Resize | Bilinear resize to fit within 224 × 224 while preserving aspect ratio |
| Padding | Center on a black 224 × 224 canvas; add borders rather than crop anatomy |
| CNN channels | Convert to RGB, repeating grayscale information across three channels |
| Tensor conversion | Convert pixel values to floating point in [0, 1] |
| Normalization | Channelwise `(x − mean) / std`, with ImageNet statistics below |

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

The cached lossless PNG contains the prepared image; tensor conversion and
normalization happen when it is loaded. The same deterministic transform is used
for validation, test and serving. Training alone may apply a mild random affine
transform: up to 5 degrees rotation, 2% translation, and scale 0.97–1.03.
This stage operates on the downloaded PNGs; it does not reconstruct raw DICOM
intensities or apply clinical windowing.

## Text preprocessing

The default configuration uses **`indication` only**. Current `findings`,
`impression`, `problems` and `mesh` never enter the predictor text.

Text cleaning lowercases content, removes HTML-like tags and standalone repeated-X
deidentification markers, and normalizes whitespace. Field prefixes distinguish
eligible text fields. Negation is retained; disease words in a genuine indication
are not stripped by default. The original text is retained for readable token
attribution and matched demonstrations.

TF-IDF uses up to 1,000 unigram/bigram features. It is fitted inside the grouped
training folds during classical-model tuning and on training data for final
models. Validation/test text is transformed without refitting. The fusion feature
scaler is also fitted on training data only.

Implementation: [`labels.py`](../src/pulmonary/labels.py),
[`text_models.py`](../src/pulmonary/text_models.py) and
[`fusion.py`](../src/pulmonary/fusion.py).

## Weak label definitions

Label version: **`indiana-weak-v1`**. The rules operate on cleaned `problems` and
`mesh` text and use case-insensitive word-boundary matching against the core term
list in [`src/pulmonary/labels.py`](../src/pulmonary/labels.py).

| Label | Exact rule | Training treatment | Report count |
| --- | --- | --- | ---: |
| Normal | Cleaned `problems` equals `normal`, and neither structured field contains a mapped core abnormal term | Target 0 | 1,379 |
| Abnormal | `problems` is not exactly `normal`, and either structured field contains a mapped core abnormal term | Target 1 | 1,222 |
| Uncertain | `problems=normal` conflicts with a core abnormal term, or neither of the above rules applies | Excluded from binary training/evaluation | 1,250 |

Mapped terms include pulmonary/pleural findings such as opacity, consolidation,
atelectasis, pneumothorax and pleural effusion, as well as cardiomegaly, mediastinal
widening and fracture. The full executable term list is authoritative. Incidental-only
annotations such as granuloma or scarring do not independently create an Abnormal
label; the Normal rule and other core terms still determine the result.

These labels are a **broad thoracic annotation proxy**, not verified pulmonary
diagnoses or the official CheXpert labeler. Uncertain is a data-curation category,
not a third class predicted by the current binary model. Excluding it changes the
evaluated population and may make performance appear better than on unfiltered
clinical cases. A Normal weak label does not establish absence of disease.

## Split definitions and leakage controls

The binary pool contains **2,601 reports and 5,068 image rows**. Uncertain reports
account for 2,398 excluded image rows. The partitions target **70% training,
15% validation and 15% test**, approximately stratified at group level, with
random seed **42**.

| Partition | Target share | Report groups | Image rows | Normal image rows | Abnormal image rows | Purpose |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Train | 70% | 1,820 | 3,544 | 1,883 | 1,661 | Fit model parameters, vectorizers and scalers; grouped CV |
| Validation | 15% | 390 | 758 | 404 | 354 | Early stopping, model selection, fusion weights and thresholds |
| Test | 15% | 391 | 766 | 408 | 358 | Final locked comparison after bundle selection/export |

Groups keep all views of one UID together. Exact image duplicates link the affected
groups before splitting. The implementation can use a complete patient identifier
if one is supplied, but this mirror's audited run uses UID. UID separation protects
against same-report leakage; it **does not independently establish patient-level
separation**. Small rounding differences and unequal numbers of views mean row
percentages need not equal the target group proportions exactly.

Prepared CSVs are `artifacts/splits/train.csv`, `validation.csv` and `test.csv`.
The manifest stores their SHA-256 hashes, checked when loading data; model registries
are tied to that manifest. Validation choices precede final test evaluation, and the
test evaluator records a lock to prevent repeated execution in the same report
directory. Label rules must not be revised to improve observed test results.

Evaluation averages image scores within each UID. The API accepts a single image
per request, so study-level metrics do not directly measure its single-view behavior.
See [data methodology](data_methodology.md) for the full evaluation design.

## Reproducibility and remaining limitations

Retain `data/download_manifest.json`, `configs/local.yaml`, the split manifest,
label/transform versions, model registries and experiment configuration with each
run. These records distinguish this mirror version and the actual experimental
population from other IU-Xray publications or exports.

Clinical indications are short and may be missing or heavily deidentified. Structured
tags are imperfect, the target includes nonpulmonary conditions, patient grouping is
unverified, and external validation is absent. The source `findings` and `impression`
remain available for provenance but were not used to train a report-generation model.
Written reports in this project summarize classifier/XAI outputs through rules.
Clinician verification is required.
