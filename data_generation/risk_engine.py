"""Deterministic risk scoring and approval logic.

This module encodes the *ground truth* that the ML models later try to learn.
It is a transparent, points-based scorecard plus a probabilistic approval rule,
mirroring how a credit officer would reason about a Cambodian micro-loan.

Separating it from the sampling code (``distributions``) means the labelling
rules can be reviewed and adjusted in one place.
"""

from __future__ import annotations

import random
from typing import Any

# Points contributed by each factor. Higher total => higher risk (0..100).
_EMPLOYMENT_RISK = {
    "Salaried": 0,
    "Self-Employed": 8,
    "Farmer": 10,
    "Student": 20,
    "Unemployed": 25,
}
_PURPOSE_RISK = {
    "House": 0,
    "Agriculture": 5,
    "SME": 8,
    "Moto": 10,
    "Car": 12,
    "Personal": 15,
}
_CREDIT_RISK = {"Good": 0, "Fair": 5, "Poor": 12, "No History": 8}
_REGION_RISK = {"Urban": 0, "Semi-Urban": 5, "Rural": 10}


def _loan_to_income_points(loan_amount: float, annual_income: float) -> int:
    if annual_income <= 0:
        return 20
    ratio = loan_amount / annual_income
    if ratio <= 1.0:
        return 0
    if ratio <= 2.0:
        return 8
    if ratio <= 3.0:
        return 15
    return 20


def _dti_points(dti: float) -> int:
    if dti <= 0.35:
        return 0
    if dti <= 0.45:
        return 5
    if dti <= 0.55:
        return 10
    return 15


def score_risk(record: dict[str, Any]) -> int:
    """Return an integer risk score in ``[0, 100]`` for an application record."""
    score = 0
    score += _EMPLOYMENT_RISK[record["EmploymentType"]]
    score += _loan_to_income_points(record["LoanAmountUSD"], record["AnnualIncomeUSD"])
    score += _PURPOSE_RISK[record["LoanType"]]
    score += _dti_points(record["DTI"])
    score += _CREDIT_RISK[record["CreditHistory"]]
    score += _REGION_RISK[record["RegionType"]]

    # Mitigating and aggravating adjustments.
    if record["CollateralUSD"] >= record["LoanAmountUSD"]:
        score -= 5
    if record["SavingsAssetsUSD"] >= record["AnnualIncomeUSD"]:
        score -= 5
    if record["LoanType"] == "Agriculture" and record["RegionType"] == "Rural":
        score -= 5
    if record["LoanType"] == "Moto" and record["EmploymentType"] in ("Salaried", "Self-Employed"):
        score -= 3
    if record["ExistingDebtUSD"] > 0.5 * record["AnnualIncomeUSD"]:
        score += 5
    if record["Age"] < 25 or record["Age"] > 60:
        score += 5

    return max(0, min(100, score))


def risk_category(score: int) -> str:
    """Bucket a numeric risk score into the three-class label."""
    if score <= 35:
        return "Low Risk"
    if score <= 65:
        return "Medium Risk"
    return "High Risk"


_BASE_APPROVAL = {"Low Risk": 0.85, "Medium Risk": 0.45, "High Risk": 0.15}


def decide_approval(rng: random.Random, record: dict[str, Any]) -> str:
    """Probabilistic approval decision conditioned on risk and key factors."""
    chance = _BASE_APPROVAL[record["RiskCategory"]]

    if record["CollateralUSD"] >= record["LoanAmountUSD"]:
        chance += 0.10
    if record["CreditHistory"] == "Good":
        chance += 0.10
    if record["EmploymentType"] == "Salaried":
        chance += 0.05
    if record["LoanType"] == "Agriculture" and record["RegionType"] == "Rural":
        chance += 0.05

    if record["DTI"] > 0.5:
        chance -= 0.10
    if record["CreditHistory"] == "Poor":
        chance -= 0.15
    if record["EmploymentType"] == "Unemployed":
        chance -= 0.20

    chance = min(0.95, max(0.05, chance))
    return "Approved" if rng.random() < chance else "Rejected"
