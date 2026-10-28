# Rule-based report generation

`src/report_generator.py` provides
`generate_report(probability, priority, gradcam_region, top_text_tokens)`.
It produces plain text with Predicted Class (Normal/Abnormal), Impression,
Visual Findings, Clinical Context and Recommendation sections.
This is deterministic template generation from the deployed multimodal model's
evidence, not a trained image-to-text model or a validated radiology report.
The API continues to require both an X-ray and clinical indication.

Both `/predict` and `/explain` return `generated_report`, `gradcam_region` and
`predicted_class`. The existing proxy label remains available for compatibility.
`/predict` computes the same attribution evidence as `/explain` but omits the
heatmap and token arrays from its response. Its latency therefore includes XAI.
The UI displays a read-only report below the results and allows text download as
`clinical_report.txt`. Reports retain "Clinician verification is required" and
contain no educational-use wording.

The impression uses the existing validation-selected review tier, rather than
introducing a new 0.5 threshold. A low score does not assert a normal examination.
Recommendations ask for radiologist review and do not prescribe clinical urgency
or follow-up from this unvalidated score.

Region descriptions summarize positive Grad-CAM activation on the preprocessed
image, divided into upper/middle/lower and image-left/central/image-right thirds.
A dimension is described as dominant only if it contains at least half the total
activation. Diffuse or absent activation is reported explicitly. Image-left and
image-right do not establish patient laterality. There is no lung segmentation,
anatomical localization or lesion detection; labels such as cardiomediastinal
silhouette would require additional validated localization evidence.

The strongest five nonzero token occlusions (excluding the deidentification token
XXXX) are reported with their actual signed contribution. Tokens are not assumed
to be symptoms, causal factors or affirmative support. Deleting text can disturb
negation and context; these descriptions remain model sensitivity evidence.

Checks: semantic report/region tests, real-model API integration, container API
requests, and in-container Streamlit rendering with an actual report response.
Container evidence: `reports/generated/functional/docker_stack_verification.json`.
