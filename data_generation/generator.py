"""Assemble synthetic loan applications into a labelled DataFrame.

``LoanDataGenerator`` ties together the sampling primitives in
``distributions`` and the labelling rules in ``risk_engine``. It is configured
entirely from ``config.yaml`` and is deterministic given the project seed.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from common import Config

from . import distributions as dist
from . import risk_engine as risk

logger = logging.getLogger(__name__)

# Column order for the emitted dataset (stable, documented in DATA_DICTIONARY).
COLUMN_ORDER = [
    "ApplicationID",
    "Province",
    "RegionType",
    "Age",
    "Gender",
    "EmploymentType",
    "AnnualIncomeUSD",
    "CreditHistory",
    "ExistingDebtUSD",
    "SavingsAssetsUSD",
    "LoanType",
    "LoanAmountUSD",
    "LoanTermYears",
    "AnnualInterestRatePct",
    "CollateralUSD",
    "DTI",
    "RiskScore",
    "RiskCategory",
    "LoanStatus",
]


class LoanDataGenerator:
    """Generate a reproducible synthetic loan-application dataset."""

    def __init__(self, config: Config):
        self.config = config
        gen = config.generation
        self.n_samples: int = int(gen["n_samples"])
        self.provinces = list(gen["provinces"].keys())
        self.province_weights = list(gen["provinces"].values())
        self.region_map: dict[str, str] = gen["region_map"]
        self.age_low, self.age_high = gen["age_range"]
        self.genders = list(gen["gender_probs"].keys())
        self.gender_weights = list(gen["gender_probs"].values())
        self.credit_labels = list(gen["credit_history_probs"].keys())
        self.credit_weights = list(gen["credit_history_probs"].values())
        # One seeded RNG drives every draw, keeping the dataset reproducible.
        self.rng = __import__("random").Random(config.seed)

    def _build_record(self, application_id: int) -> dict:
        rng = self.rng
        province = rng.choices(self.provinces, weights=self.province_weights)[0]
        region_type = self.region_map[province]
        age = rng.randint(self.age_low, self.age_high)
        gender = rng.choices(self.genders, weights=self.gender_weights)[0]

        employment_type = dist.pick_employment_type(rng, region_type)
        annual_income = dist.pick_annual_income(rng, age, region_type, employment_type)
        credit_history = rng.choices(self.credit_labels, weights=self.credit_weights)[0]

        # Existing debt: 35% of applicants carry some, up to 1.2x income.
        if rng.choices([True, False], weights=[0.35, 0.65])[0]:
            existing_debt = rng.randint(100, max(100, int(annual_income * 1.2)))
        else:
            existing_debt = 0
        savings = rng.randint(0, max(100, int(annual_income * 2)))

        loan_type = dist.pick_loan_type(rng, region_type, employment_type)
        loan_amount = dist.pick_loan_amount(rng, loan_type, annual_income, region_type)
        loan_term = dist.pick_loan_term(rng, loan_type)
        interest_rate = dist.pick_interest_rate(rng, loan_type)
        collateral = dist.pick_collateral(rng, loan_type, loan_amount)

        payment = dist.annual_debt_service(loan_amount, interest_rate, loan_term)
        # Cap DTI at a finite sentinel so zero-income rows stay model-safe.
        dti = round(payment / annual_income, 3) if annual_income > 0 else 99.0
        dti = min(dti, 99.0)

        record = {
            "ApplicationID": application_id,
            "Province": province,
            "RegionType": region_type,
            "Age": age,
            "Gender": gender,
            "EmploymentType": employment_type,
            "AnnualIncomeUSD": annual_income,
            "CreditHistory": credit_history,
            "ExistingDebtUSD": existing_debt,
            "SavingsAssetsUSD": savings,
            "LoanType": loan_type,
            "LoanAmountUSD": loan_amount,
            "LoanTermYears": loan_term,
            "AnnualInterestRatePct": interest_rate,
            "CollateralUSD": collateral,
            "DTI": dti,
        }

        score = risk.score_risk(record)
        record["RiskScore"] = score
        record["RiskCategory"] = risk.risk_category(score)
        record["LoanStatus"] = risk.decide_approval(rng, record)
        return record

    def generate(self) -> pd.DataFrame:
        """Build the full dataset as a DataFrame."""
        logger.info("Generating %d synthetic loan applications...", self.n_samples)
        records = [self._build_record(i) for i in range(1, self.n_samples + 1)]
        df = pd.DataFrame(records)[COLUMN_ORDER]
        logger.info("Generation complete: %d rows, %d columns", *df.shape)
        return df

    def save(self, df: pd.DataFrame, path: str | Path | None = None) -> Path:
        """Persist the dataset to CSV and return the resolved path."""
        out = Path(path) if path else self.config.path(self.config.generation["output_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        logger.info("Saved dataset to %s", out)
        return out

    @staticmethod
    def summarise(df: pd.DataFrame) -> str:
        """Return a short human-readable summary of the generated dataset."""
        approval_rate = (df["LoanStatus"] == "Approved").mean() * 100
        lines = [
            f"Total applications : {len(df):,}",
            f"Approval rate      : {approval_rate:.1f}%",
            "Risk distribution  :",
        ]
        risk_counts = df["RiskCategory"].value_counts()
        for label in ["Low Risk", "Medium Risk", "High Risk"]:
            count = int(risk_counts.get(label, 0))
            pct = count / len(df) * 100 if len(df) else 0
            lines.append(f"    {label:<12}: {count:>7,} ({pct:4.1f}%)")
        return "\n".join(lines)
