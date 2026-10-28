# Local experiment artifacts

Generated files are excluded from git. The expected structure is:

```text
splits/           CSVs, uncertain cases and audited manifest
image_cache/      deterministic lossless images for efficient training
models/           trained estimators, CNN/fusion checkpoints and registries
embeddings/       ordered image embeddings with sample IDs
bundle/           selected serving files + checksummed manifest
mlflow.db         local tracking database
optuna.db         tuning study
metadata-only/    provisional text experiments with unverified image linkage
team_classification/  reconstructed team cohort, checkpoints, predictions, separate MLflow DB
team_bundle/      checksummed ResNet50 + DenseNet121 + text serving bundle
team_import/      reviewed learned-report exports and source manifests
```

A reference v1 bundle includes `manifest.json` and `backbone.pt`; early fusion additionally requires `fusion.pt`, `vectorizer.joblib`, and `scaler.joblib`; v1 late fusion additionally requires `text.joblib`.

The current `team_bundle/` instead contains `manifest.json`, `resnet50.pt`,
`densenet121.pt` and `text_model.joblib`. Its preprocessing, cohort and threshold
are versioned separately. The API verifies listed file checksums and preprocessing
identity before serving. Restore only trusted complete bundles.
