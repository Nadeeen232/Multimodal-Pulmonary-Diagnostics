# Final SIC submission checklist

Source: SIC8 AI Project Final handbook, page 17, “The complete submission package”.
Current model: **0.2.0-team29-47e281fc**. Technical package: **8/8 items assembled
and verified locally**. Saved operational evidence verified.

| Official SIC item | Status | Primary artifacts |
| --- | --- | --- |
| 01 Source code | Complete | `src/pulmonary/team_classification.py`; `src/pulmonary/team_inference.py`; `app/main.py`; `ui/app.py` |
| 02 Technical documentation | Complete | `README.md`; `pyproject.toml`; `compose.yaml`; `Dockerfile`; `docs/architecture.png`; `docs/demo_runbook.md` |
| 03 Data assets | Complete | `docs/data_assets.md`; `docs/data_methodology.md`; `artifacts/team_classification/cohort_manifest.json` |
| 04 Technical evidence | Complete | `docs/technical_evidence_report.md`; `docs/technical_evidence_report.docx`; `reports/evidence/team29/ablation_results.csv`; `reports/evidence/team29/pytest.txt`; `reports/evidence/mlflow_dashboard.png`; `reports/evidence/mlflow_presentation.png`; `reports/evidence/team29/mlflow_tracking_snapshot.json`; `reports/evidence/tracking_snapshot.json`; `reports/evidence/optuna_trials.csv`; `reports/evidence/optuna_best_params.json`; `dist/SIC_0.2.0-team29_tracking_evidence.zip` |
| 05 Working solution | Complete | `artifacts/team_bundle/manifest.json`; `dist/SIC_0.2.0-team29_model_bundle.zip`; `reports/evidence/team29/deployment_verification.json` |
| 06 Presentation deck | Complete | `docs/SIC_Final_Presentation.pptx`; `docs/presentation_outline.md`; `scripts/build_presentation.py` |
| 07 Demonstration | Complete | `docs/demo_runbook.md`; `docs/demo_fallback.webm`; `scripts/automated_demo.py`; `demo_images/demo_cases.csv` |
| 08 Team contribution summary | Complete | `team_contributions.md` |

## Handoff requirements

Submit the source repository plus **both ZIPs in `dist/`**. Model binaries,
tracking databases and raw data are excluded from Git. A source-only checkout
will not predict until its model archive is restored. The separate tracking ZIP
contains a portable COPY of the current database and referenced run artifacts;
the original database is unchanged. Dataset references satisfy the data-assets
item; redistribution of the full dataset is not required.

The deck has 13 slides and updated notes; the Word report contains 18
embedded figures and 13 editable tables. A real silent fallback video
was recorded and decoded successfully. Photos and a repository QR code are not
required by the official checklist; the deck uses member initials and no invented URL.
All PowerPoint files remain excluded from GitHub; submit the generated deck
separately. The additional experiment-tracking slide preserves the original
project content and distinguishes current MLflow runs from historical Optuna tuning.

## Accurate claims and human readiness

Current fusion is 82.87% accuracy, 0.7918 F1, 0.8816 ROC-AUC on 356 studies.
The report-generator finding is documented OpenI dataset-source pretraining overlap,
not proof of individual test-image membership. The deterministic report remains
the default. No external clinical validation or measured workflow benefit is claimed.

The team still needs its own rehearsal and final review of attribution. Automated
artifact checks cannot establish individual understanding. GitHub publication was
not performed. `docs/submission_manifest.json` records versioned checksums for handoff.
