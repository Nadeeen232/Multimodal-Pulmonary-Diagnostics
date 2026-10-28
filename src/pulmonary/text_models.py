"""Text models with group-safe CV and fold-local TF-IDF fitting."""

from pathlib import Path

import joblib
import mlflow
import numpy as np
import optuna
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from pulmonary.common import load_splits, save_json, split_digest
from pulmonary.evaluation import choose_threshold, cost_evidence, evidence
from pulmonary.tracking import log_results, tracked


def text_pipeline(classifier, config, max_features=None):
    return Pipeline(
        [
            # Preserve negation; sklearn's english stop list removes "no"/"not".
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features or config["max_features"],
                    ngram_range=(1, 2),
                    stop_words=None,
                    min_df=1,
                    sublinear_tf=True,
                ),
            ),
            ("classifier", classifier),
        ]
    )


def train_text(config):
    splits = load_splits(config)
    train = splits["train"].drop_duplicates("uid").reset_index(drop=True)
    val = splits["validation"]
    x, y = train.model_text, train.target
    if not x.str.strip().ne("").any():
        raise ValueError("Training text is empty")
    fold_list = list(
        StratifiedGroupKFold(
            n_splits=config["cv_folds"], shuffle=True, random_state=config["seed"]
        ).split(x, y, train.group_id)
    )
    for a, b in fold_list:
        if set(y.iloc[a]) != {0, 1} or set(y.iloc[b]) != {0, 1}:
            raise ValueError("Too few independent class groups for requested CV folds")
    model_root = Path(config["output_root"]) / "models"
    model_root.mkdir(parents=True, exist_ok=True)
    report_root = Path(config["reports_root"]) / "text"
    report_root.mkdir(parents=True, exist_ok=True)
    candidates = {}
    candidates["text_dummy"] = text_pipeline(DummyClassifier(strategy="prior"), config)
    with tracked(config, "linear", "logistic-optuna"):
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=config["seed"]),
            storage=f"sqlite:///{Path(config['output_root']) / 'optuna.db'}",
            study_name="text-logistic",
            load_if_exists=False,
        )

        def objective(trial):
            c = trial.suggest_float("C", 1e-3, 100, log=True)
            features = trial.suggest_categorical(
                "max_features", sorted(set([500, config["max_features"], 2000]))
            )
            pipeline = text_pipeline(
                LogisticRegression(
                    C=c, class_weight="balanced", max_iter=2000, random_state=config["seed"]
                ),
                config,
                features,
            )
            scores = cross_val_score(
                pipeline, x, y, cv=fold_list, scoring="roc_auc", n_jobs=1, error_score="raise"
            )
            with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
                mlflow.log_params(trial.params)
                mlflow.log_metric("cv_roc_auc", float(scores.mean()))
            return float(scores.mean())

        study.optimize(objective, n_trials=config["optuna_trials"])
        study.trials_dataframe().to_csv(report_root / "optuna_trials.csv", index=False)
        save_json(report_root / "optuna_best_params.json", study.best_params)
        candidates["text_logistic"] = text_pipeline(
            LogisticRegression(
                C=study.best_params["C"],
                class_weight="balanced",
                max_iter=2000,
                random_state=config["seed"],
            ),
            config,
            study.best_params["max_features"],
        )
        mlflow.log_metric("best_cv_roc_auc", study.best_value)
        mlflow.log_artifact(str(report_root / "optuna_trials.csv"))
    # Calibration folds are explicitly grouped. The entire TF-IDF pipeline is refit per fold.
    svm = text_pipeline(
        LinearSVC(C=1.0, class_weight="balanced", random_state=config["seed"]), config
    )
    candidates["text_svm"] = CalibratedClassifierCV(svm, cv=fold_list, method="sigmoid")
    candidates["text_extra_trees"] = text_pipeline(
        ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            class_weight="balanced",
            max_features="sqrt",
            n_jobs=-1,
            random_state=config["seed"],
        ),
        config,
    )
    registry = {}
    for name, estimator in candidates.items():
        family = {"text_extra_trees": "tree_ensemble", "text_dummy": "dummy"}.get(name, "linear")
        with tracked(config, family, name, {"class_weight": "balanced"}):
            estimator.fit(x, y)
            probs = estimator.predict_proba(val.model_text)[:, 1]
            threshold, table = choose_threshold(val, probs, config)
            folder = report_root / name
            result = evidence(val, probs, threshold, folder, "validation")
            cost_evidence(table, folder)
            path = model_root / f"{name}.joblib"
            joblib.dump(estimator, path)
            np.savez(
                model_root / f"{name}_validation.npz",
                sample_id=val.sample_id.to_numpy(dtype=str),
                probability=probs,
            )
            log_results(result, folder, path)
            registry[name] = {
                "name": name,
                "family": family,
                "kind": "text",
                "artifact": path.name,
                "threshold": threshold,
                "validation": result,
            }
    best = max(
        (name for name in registry if name != "text_dummy"),
        key=lambda name: registry[name]["validation"]["roc_auc"],
    )
    save_json(
        model_root / "text_registry.json",
        {"best": best, "models": registry, "split_manifest_sha256": split_digest(config)},
    )
    return registry
