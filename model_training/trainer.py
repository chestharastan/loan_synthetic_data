"""Train, evaluate and persist the loan risk & approval models.

Both targets (multiclass risk category and binary approval) are learned by
independent RandomForest pipelines that share the same preprocessing and the
same feature set, so they can be served side by side from one input record.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from common import Config

from .evaluate import evaluate_classifier, text_report
from .preprocessing import build_preprocessor, feature_columns

logger = logging.getLogger(__name__)


@dataclass
class TrainedTarget:
    """Result bundle for one trained model."""

    name: str
    pipeline: Pipeline
    metrics: dict[str, Any]
    report: str
    model_path: Path


class ModelTrainer:
    """Fit and evaluate the risk and approval models from the dataset."""

    def __init__(self, config: Config):
        self.config = config
        self.features = feature_columns(config)

    # -- internals -----------------------------------------------------------
    def _make_pipeline(self) -> Pipeline:
        params = self.config.training["model"]
        return Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(self.config)),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=params["n_estimators"],
                        max_depth=params["max_depth"],
                        min_samples_leaf=params["min_samples_leaf"],
                        class_weight=params["class_weight"],
                        n_jobs=params["n_jobs"],
                        random_state=self.config.seed,
                    ),
                ),
            ]
        )

    def _train_one(
        self,
        df: pd.DataFrame,
        name: str,
        target_col: str,
        labels: list | None,
        display_names: list[str] | None = None,
    ) -> TrainedTarget:
        logger.info("Training '%s' model on target '%s'...", name, target_col)
        X = df[self.features]
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.training["test_size"],
            random_state=self.config.seed,
            stratify=y,
        )

        pipeline = self._make_pipeline()
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        metrics = evaluate_classifier(
            y_test, y_pred, labels=labels, display_names=display_names
        )
        report = text_report(
            f"{name} model", y_test, y_pred, labels=labels, display_names=display_names
        )

        model_dir = self.config.path(self.config.training["model_dir"])
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"{name}_pipeline.joblib"
        dump(pipeline, model_path)
        logger.info("Saved %s model to %s (accuracy=%.3f)",
                    name, model_path, metrics["accuracy"])

        return TrainedTarget(name, pipeline, metrics, report, model_path)

    # -- public API ----------------------------------------------------------
    def run(self, df: pd.DataFrame | None = None) -> dict[str, TrainedTarget]:
        """Train both targets, persist models + metrics, return the results."""
        if df is None:
            input_path = self.config.path(self.config.training["input_path"])
            logger.info("Loading dataset from %s", input_path)
            df = pd.read_csv(input_path)

        targets = self.config.training["targets"]
        results: dict[str, TrainedTarget] = {}

        # Risk: multiclass with an explicit label order (low -> high).
        results["risk"] = self._train_one(
            df, "risk", targets["risk"]["column"], labels=targets["risk"]["labels"]
        )
        # Approval: binary, encoded to 0/1 with positive = "Approved".
        approval = targets["approval"]
        y_binary = (df[approval["column"]] == approval["positive_label"]).astype(int)
        df_bin = df.assign(_approval=y_binary)
        results["approval"] = self._train_one(
            df_bin, "approval", "_approval",
            labels=[0, 1], display_names=["Rejected", "Approved"],
        )

        self._write_metrics(results)
        return results

    def _write_metrics(self, results: dict[str, TrainedTarget]) -> None:
        metrics_dir = self.config.path(self.config.training["metrics_dir"])
        metrics_dir.mkdir(parents=True, exist_ok=True)
        payload = {name: r.metrics for name, r in results.items()}
        out = metrics_dir / "metrics.json"
        with out.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        logger.info("Wrote evaluation metrics to %s", out)
