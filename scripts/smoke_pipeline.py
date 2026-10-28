"""Exercise every stage on explicit artificial data; no clinical quality claims.

All generated files live in a new .tmp directory. Downloaded ImageNet weights
are real pretrained weights; the artificial-data training is a software check.
"""

import json
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fastapi.testclient import TestClient
from PIL import Image

from app.main import create_app
from pulmonary.artifacts import evaluate_test, explanation_examples, export_bundle
from pulmonary.cli import report
from pulmonary.common import load_splits, read_config
from pulmonary.data import prepare
from pulmonary.fusion import train_fusion
from pulmonary.image_models import train_images
from pulmonary.inference import InferenceEngine
from pulmonary.text_models import train_text


def main():
    torch.set_num_threads(2)
    root = Path(".tmp") / f"software-smoke-{uuid.uuid4().hex[:8]}"
    raw = root / "raw"
    raw.mkdir(parents=True)
    reports, projections = [], []
    for index in range(40):
        label = "normal" if index % 2 == 0 else "pneumonia"
        reports.append(
            {
                "uid": str(index),
                "problems": label,
                "mesh": label,
                "indication": "no cough" if index % 2 == 0 else "persistent cough",
                "comparison": "",
            }
        )
        for view in [0, 1]:
            filename = f"case_{index}_{view}.png"
            array = np.random.default_rng(index * 2 + view).integers(
                0, 256, size=(48, 80), dtype=np.uint8
            )
            Image.fromarray(array).save(raw / filename)
            projections.append({"uid": str(index), "filename": filename})
    pd.DataFrame(reports).to_csv(raw / "indiana_reports.csv", index=False)
    pd.DataFrame(projections).to_csv(raw / "indiana_projections.csv", index=False)
    config = read_config()
    config.update(
        data_root=str(raw),
        output_root=str(root / "artifacts"),
        reports_root=str(root / "reports"),
        tracking_uri=f"sqlite:///{root / 'mlflow.db'}",
        image_size=64,
        batch_size=8,
        optuna_trials=2,
        cv_folds=2,
        head_epochs=1,
        finetune_epochs=1,
        fusion_epochs=2,
    )
    prepare(config)
    train_text(config)
    train_images(config)
    train_fusion(config)
    export_bundle(config)
    evaluate_test(config)
    explanation_examples(config, limit=2)
    report(config)
    engine = InferenceEngine(root / "artifacts/bundle")
    frame = load_splits(config)["validation"]
    row = frame.iloc[0]
    image = (raw / row.image_path).read_bytes()
    with TestClient(create_app(lambda: engine)) as client:
        response = client.post(
            "/explain",
            files={"image": ("fixture.png", image, "image/png")},
            data={"indication": row.indication},
        )
        assert response.status_code == 200, response.text
        assert response.json()["gradcam_base64"]
    (root / "SOFTWARE_CHECK_ONLY.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "dataset": "artificial software fixture",
                "clinical_evidence": False,
            },
            indent=2,
        )
    )
    print(f"SOFTWARE PIPELINE CHECK PASSED: {root}", flush=True)


if __name__ == "__main__":
    main()
