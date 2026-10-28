# System architecture

## Current Team29 profile

The deployed bundle is **0.2.0-team29-47e281fc**. `team_classification.py` prepares
the strict-label cohort and trains the selected approach. `team_features.py`
shares exact 384 × 384 grayscale Lanczos letterboxing and ImageNet normalization
between training and serving. Indication/comparison text becomes word/character
TF-IDF plus train-scaled metadata features and logistic-regression scores.

The image branch averages ResNet50 and DenseNet121 scores. Validation selects
70% image / 30% text fusion and a balanced-accuracy threshold of 0.49. Frozen
choices are evaluated on test before the versioned bundle is exported.
`team_inference.py` verifies hashes and preprocessing identity; `load_engine`
dispatches by bundle kind. Study evaluation averages views, while the UI analyzes
one image at a time.

Grad-CAM targets the ResNet50 branch of the final fused score. Raw-token occlusion
measures text associations with the image scores fixed. The default written
report remains deterministic. The reviewed FLAN-T5 adapter is an experimental
option outside the default service; see [technical_evidence_report.md](technical_evidence_report.md).

## Reference v1 workflow

The source workflow is: dataset CSVs/images → validated pairing and grouped splits → tracked comparative models → validation selection → checksummed multimodal bundle → FastAPI → Streamlit.

`src/pulmonary/data.py` audits data and creates lossless square inputs by grayscale conversion, aspect-preserving containment, and zero borders. `images.py` applies this same deterministic operation and ImageNet normalization during serving. Mild augmentation is restricted to training.

Text training uses grouped CV around the whole TF-IDF pipeline. Image training compares ResNet50 and DenseNet121. Early fusion combines frozen CNN embeddings and train-only TF-IDF, standardizes features using training statistics, and applies a small MLP. Late fusion averages probabilities with a validation-selected nonzero weight for each modality. Validation ROC-AUC selects the final multimodal candidate; cost sensitivity selects its threshold.

The API loads its immutable bundle once at startup. Checksums and transform versions must match. A shared lock serializes model calls and Grad-CAM hooks. Inference work runs in a thread pool. `/health` reports process and model state; `/ready` returns 503 until a model is usable. The UI communicates through HTTP and does not load weights.

Grad-CAM differentiates the final deployed multimodal score with the current text held fixed. Text occlusion removes one raw-note token at a time and reruns the same multimodal model with the image held fixed. Explanations describe sensitivity, not diagnosis or causation.

See `architecture.svg` for the exported diagram. Container readiness and trained-model performance remain separately verifiable requirements.
