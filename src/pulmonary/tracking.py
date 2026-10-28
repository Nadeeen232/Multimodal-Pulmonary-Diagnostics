"""Local MLflow experiments with input/config hashes and real artifacts."""

from contextlib import contextmanager
from pathlib import Path

import mlflow

from pulmonary.common import file_hash, revision, save_json, source_digest


@contextmanager
def tracked(config, family, name, params=None):
    mlflow.set_tracking_uri(config["tracking_uri"])
    mlflow.set_experiment(f"sic-{family}")
    with mlflow.start_run(run_name=name):
        mlflow.set_tag("source_sha256", source_digest())
        mlflow.set_tags(
            {"git_revision": revision(), "family": family, "educational_prototype": "true"}
        )
        mlflow.log_params({"seed": config["seed"], **(params or {})})
        path = Path(config["reports_root"]) / "run_config.json"
        save_json(path, config)
        mlflow.log_artifact(str(path))
        manifest = Path(config["output_root"]) / "splits/manifest.json"
        if manifest.exists():
            mlflow.log_param("split_manifest_sha256", file_hash(manifest))
            mlflow.log_artifact(str(manifest))
        yield


def log_results(result, folder, model_path):
    mlflow.log_metrics(
        {
            f"validation_{key}": value
            for key, value in result.items()
            if isinstance(value, (float, int))
        }
    )
    mlflow.log_artifacts(str(folder), artifact_path="validation_evidence")
    mlflow.log_artifact(str(model_path), artifact_path="model")
