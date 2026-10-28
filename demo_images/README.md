# Versioned demonstration cases

Use **`team29/`** for the current deployed model. The exact image, indication,
comparison, model score and weak-label outcome are listed in `demo_cases.csv`.
The corresponding `_context.txt` files make the text easy to copy into Streamlit.

| Case | File | Weak label | Current prediction |
| --- | --- | --- | --- |
| Normal example | `team29/01_true_negative_uid_3212.png` | Normal | Normal |
| Abnormal example | `team29/02_true_positive_uid_2348.png` | Abnormal | Abnormal |
| False-positive example | `team29/03_false_positive_uid_2829.png` | Normal | Abnormal |

These three cases were selected after held-out evaluation to demonstrate the UI.
They are not an unbiased performance sample. Scores here are for one image,
whereas the ablation table averages available views per study.

The older root images and `demo_cases_v1.csv` remain as reference assets. The
automated UID 1330 demo still runs with its original image/indication and is Normal
under the new model, with approximately 18.54% score when comparison is omitted.
Use the current CSV for current-model demonstrations rather than inferring outcomes
from the historical root filenames.
