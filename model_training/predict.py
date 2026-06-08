"""Inference: load the trained pipelines and score new loan applications.

``LoanScorer`` wraps the two persisted joblib pipelines behind a single
``score()`` call, which is the natural integration point for a web backend
(e.g. the Django Loan-Advisor app) or a batch job.

Run directly for a quick smoke test:
    python -m model_training.predict
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from joblib import load

from common import Config, load_config

from .preprocessing import feature_columns

logger = logging.getLogger(__name__)


class LoanScorer:
    """Load saved pipelines and predict risk category + approval status."""

    def __init__(self, config: Config):
        self.config = config
        self.features = feature_columns(config)
        model_dir = config.path(config.training["model_dir"])
        self.risk_model = self._load(model_dir / "risk_pipeline.joblib")
        self.approval_model = self._load(model_dir / "approval_pipeline.joblib")

    @staticmethod
    def _load(path: Path):
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found: {path}. Run `python -m model_training.train` first."
            )
        return load(path)

    def score(self, application: dict[str, Any]) -> dict[str, Any]:
        """Score a single application dict and return predictions."""
        frame = pd.DataFrame([application])[self.features]
        risk_pred = self.risk_model.predict(frame)[0]
        approval_pred = int(self.approval_model.predict(frame)[0])
        return {
            "risk_category": str(risk_pred),
            "loan_status": "Approved" if approval_pred == 1 else "Rejected",
        }

    def score_batch(self, applications: pd.DataFrame) -> pd.DataFrame:
        """Score many applications; returns the input plus prediction columns."""
        frame = applications[self.features]
        out = applications.copy()
        out["PredictedRiskCategory"] = self.risk_model.predict(frame)
        out["PredictedLoanStatus"] = [
            "Approved" if p == 1 else "Rejected"
            for p in self.approval_model.predict(frame)
        ]
        return out


def _demo() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    config = load_config()
    scorer = LoanScorer(config)

    sample = {
        "Province": "Kandal",
        "RegionType": "Semi-Urban",
        "Age": 43,
        "Gender": "Female",
        "EmploymentType": "Salaried",
        "AnnualIncomeUSD": 9000,
        "CreditHistory": "Good",
        "ExistingDebtUSD": 0,
        "SavingsAssetsUSD": 6550,
        "LoanType": "House",
        "LoanAmountUSD": 10775,
        "LoanTermYears": 10,
        "AnnualInterestRatePct": 14.23,
        "CollateralUSD": 12000,
        "DTI": 0.18,
    }
    print("Input application:")
    for k, v in sample.items():
        print(f"  {k}: {v}")
    print("\nPrediction:", scorer.score(sample))


if __name__ == "__main__":
    _demo()
