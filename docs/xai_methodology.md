# Explanation methodology

## Current Team29 release

The 0.2.0 late-fusion bundle uses Grad-CAM at the ResNet50 branch of the final
fused score. It does not return a combined DenseNet/ResNet anatomical map.
Text attribution is raw-token occlusion over indication/comparison, not SHAP.
Current figures come from `scripts/build_submission_assets.py` and are saved
under `reports/evidence/team29/`, with exact deployed model versions and inputs.
The three selected test examples illustrate a true negative, true positive and
false positive against weak labels. They do not estimate performance.

## Shared principles and historical v1 implementation

Image attribution uses Grad-CAM on the encoder's final feature map. ResNet50 exposes its last residual stage; DenseNet121 exposes its post-ReLU dense features. Gradients are taken from the deployed fusion model's final abnormality score, conditioned on the current text. The spatially averaged gradients weight feature channels, followed by ReLU, per-example normalization and bilinear resizing. The map is overlaid on the same aspect-preserving prepared image that the model sees.

For early fusion, differentiation flows through image pooling, feature concatenation, the saved training scaler, and the fusion MLP. For late fusion, it flows through the image score multiplied by the deployed image weight. This avoids presenting a separate image-only classifier's attribution as an explanation of the multimodal decision.

Text attribution uses leave-one-token-out occlusion on the actual raw indication/comparison. Each alteration goes through the same text preprocessing and saved model. Contribution = full-input score minus score with that token removed. Positive values mean that token's presence raises the abnormality score in this context. These changes are not additive because TF-IDF normalization, n-grams, and the fusion network introduce interactions. Positions refer to the submitted raw string. At most 40 eligible tokens are examined and truncation is reported explicitly.

Grad-CAM's magnitude is normalized separately for each case, so map color intensity must not be compared between patients. Diffuse or empty maps, frequent words, shortcut features, and padding focus are possible. Neither image regions nor text contributions validate a diagnosis or prove causation.

`pulmonary explain-examples` uses one image per validation study, matching the API, and saves PNG overlays, JSON predictions/contributions, and combined Markdown case cards. The default limit is 20, aiming for five each of TN/TP/FP/FN against weak labels. If a category lacks enough cases, other cases fill the quota. `selection_manifest.json` records actual availability and saved counts; missing categories are never fabricated. The example selection and failure modes must be reviewed for the final presentation.
