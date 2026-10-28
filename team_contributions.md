# Team responsibilities and contributions

Current release: **0.2.0-team29-47e281fc**. This records responsibilities and
contributions reported by the project owner, supported by shared artifacts.
It is not an inferred commit or hours ledger.

| Member | Technical responsibility | Contribution / component | Supporting evidence |
| --- | --- | --- | --- |
| Martina | Data Preparation & EDA with Yousef | Data-preparation and exploratory-analysis workstream; schema, pairing, labels and data overview | Data Assets, data audit/EDA outputs and split manifests |
| Yousef | Data Preparation & EDA with Martina | Data-preparation and exploratory-analysis workstream; provenance, missing data, splits and leakage controls | Data methodology, Data Assets and cohort manifest |
| Zeyad | Deep Learning & Multimodal Fusion with Nadeen | Provided the model notebooks and image/text classification and fusion approach for integration | Supplied notebooks and docs/technical_evidence_report.md |
| Nadeen | Deep Learning & Multimodal Fusion with Zeyad | Provided the model notebooks and experimental evidence/exports for architecture and report-generation review | Supplied notebooks, report-generator exports and metrics/run comparisons |
| Mohamed Ehab | Integration, evaluation, FastAPI, Docker, Streamlit, XAI, reporting and presentation | Integrated the notebook approach into Python files; adapted batch-8 AMP training; ran evaluation; exported/served the bundle; verified API/UI; documented report-generator overlap | src/pulmonary/team_*.py, app/, ui/, reports/evidence/team29/, current report and deck |

## Notebook audit and reporting decision

Mohamed Ehab traced the report generator's XRV tag model to a weight configuration
that explicitly includes OpenI in pretraining. Indiana IU comes from the OpenI/
NLMCXR collection. The finding establishes **dataset-source pretraining overlap
and an unresolved evaluation-leakage risk**. Available files do not prove that
particular local test images or patients were in pretraining; no image-level
leakage proof is claimed.

The audit also recorded missed findings and distinguished two different V2 runs.
The deterministic report remains the default; the learned adapter is experimental.
The deployed classifier uses ImageNet weights and does not use that XRV component.

## Attribution and handoff

Implementation, analysis and document generation were AI-assisted. This summary
identifies human project roles and supplied work, not exclusive human authorship
of generated code. Individual hours and commit totals were not supplied and are
not invented. Each member should rehearse their component and its interfaces
using docs/team_study_guide.md.
