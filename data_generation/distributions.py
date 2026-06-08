"""Domain sampling primitives for Cambodian loan applications.

Each function encapsulates one piece of business knowledge (how income scales
with region and employment, what loan sizes are realistic, what terms and
interest rates apply per product). Keeping them small and pure makes the
generation logic easy to audit and to unit-test.

All randomness flows through an injected ``random.Random`` instance so the
whole pipeline is reproducible from a single seed.
"""

from __future__ import annotations

import random

# --- Loan product catalogue --------------------------------------------------
LOAN_TYPES = ["House", "Car", "Moto", "Personal", "SME", "Agriculture"]

# Annual income ranges (USD) by region and employment type.
INCOME_RANGES: dict[str, dict[str, tuple[int, int]]] = {
    "Urban": {
        "Salaried": (3600, 24000),
        "Self-Employed": (2400, 18000),
        "Farmer": (1800, 6000),
        "Student": (600, 2400),
        "Unemployed": (0, 1200),
    },
    "Semi-Urban": {
        "Salaried": (2400, 12000),
        "Self-Employed": (1800, 9000),
        "Farmer": (1500, 4800),
        "Student": (600, 1800),
        "Unemployed": (0, 900),
    },
    "Rural": {
        "Salaried": (1800, 7200),
        "Self-Employed": (1200, 4800),
        "Farmer": (1200, 3600),
        "Student": (300, 1200),
        "Unemployed": (0, 600),
    },
}

# Employment-type mix by region (weights align with the labels list).
EMPLOYMENT_LABELS = ["Salaried", "Self-Employed", "Farmer", "Student", "Unemployed"]
EMPLOYMENT_WEIGHTS: dict[str, list[float]] = {
    "Urban": [0.55, 0.30, 0.05, 0.05, 0.05],
    "Semi-Urban": [0.40, 0.35, 0.15, 0.05, 0.05],
    "Rural": [0.25, 0.10, 0.50, 0.10, 0.05],
}

# Loan-amount ranges (USD) by product and region.
LOAN_AMOUNT_RANGES: dict[str, dict[str, tuple[int, int]]] = {
    "House": {"Urban": (5000, 50000), "Semi-Urban": (3000, 30000), "Rural": (2000, 20000)},
    "Car": {"Urban": (3000, 20000), "Semi-Urban": (2000, 15000), "Rural": (1500, 10000)},
    "Moto": {"Urban": (500, 3000), "Semi-Urban": (400, 2500), "Rural": (300, 2000)},
    "Personal": {"Urban": (300, 5000), "Semi-Urban": (200, 3000), "Rural": (100, 2000)},
    "SME": {"Urban": (2000, 30000), "Semi-Urban": (1500, 20000), "Rural": (1000, 15000)},
    "Agriculture": {"Urban": (500, 10000), "Semi-Urban": (500, 15000), "Rural": (300, 20000)},
}

# Loan term (years) and annual interest rate (%) ranges per product.
LOAN_TERMS: dict[str, tuple[int, int]] = {
    "House": (5, 25),
    "Car": (2, 6),
    "Moto": (1, 3),
    "Personal": (1, 4),
    "SME": (1, 7),
    "Agriculture": (1, 3),
}
INTEREST_RATES: dict[str, tuple[float, float]] = {
    "House": (9.6, 18.0),
    "Car": (12.0, 18.0),
    "Moto": (24.0, 36.0),
    "Personal": (18.0, 30.0),
    "SME": (18.0, 30.0),
    "Agriculture": (21.6, 36.0),
}

# Loan-product preference weights by region and employment segment.
_LOAN_TYPE_WEIGHTS: dict[str, dict[str, list[float]]] = {
    # order matches LOAN_TYPES: House, Car, Moto, Personal, SME, Agriculture
    "Urban": {
        "Farmer": [0.05, 0.05, 0.30, 0.20, 0.10, 0.30],
        "Worker": [0.25, 0.15, 0.20, 0.20, 0.15, 0.05],
        "Other": [0.05, 0.05, 0.40, 0.40, 0.05, 0.05],
    },
    "Semi-Urban": {
        "Farmer": [0.05, 0.05, 0.25, 0.15, 0.10, 0.40],
        "Worker": [0.20, 0.10, 0.25, 0.20, 0.15, 0.10],
        "Other": [0.05, 0.05, 0.45, 0.35, 0.05, 0.05],
    },
    "Rural": {
        "Farmer": [0.05, 0.02, 0.20, 0.13, 0.10, 0.50],
        "Worker": [0.15, 0.05, 0.30, 0.20, 0.15, 0.15],
        "Other": [0.05, 0.02, 0.48, 0.35, 0.05, 0.05],
    },
}


def pick_employment_type(rng: random.Random, region_type: str) -> str:
    """Sample an employment type conditioned on the region."""
    return rng.choices(EMPLOYMENT_LABELS, weights=EMPLOYMENT_WEIGHTS[region_type])[0]


def pick_annual_income(
    rng: random.Random, age: int, region_type: str, employment_type: str
) -> int:
    """Sample annual income (USD), nudged upward for prime working age."""
    low, high = INCOME_RANGES[region_type][employment_type]
    if 25 <= age <= 45:
        high = int(high * 1.2)
    elif age > 45:
        high = int(high * 0.9)
    return rng.randint(low, max(low, high))


def pick_loan_type(rng: random.Random, region_type: str, employment_type: str) -> str:
    """Sample a loan product conditioned on region and employment segment."""
    if employment_type == "Farmer":
        segment = "Farmer"
    elif employment_type in ("Salaried", "Self-Employed"):
        segment = "Worker"
    else:
        segment = "Other"
    weights = _LOAN_TYPE_WEIGHTS[region_type][segment]
    return rng.choices(LOAN_TYPES, weights=weights)[0]


def pick_loan_amount(
    rng: random.Random, loan_type: str, annual_income: int, region_type: str
) -> int:
    """Sample a loan amount (USD), capped at 5x annual income for realism."""
    min_amt, max_amt = LOAN_AMOUNT_RANGES[loan_type][region_type]
    max_amt = min(max_amt, annual_income * 5)
    max_amt = max(max_amt, min_amt)
    return rng.randint(min_amt, max_amt)


def pick_loan_term(rng: random.Random, loan_type: str) -> int:
    """Sample a loan term in years for the product."""
    lo, hi = LOAN_TERMS[loan_type]
    return rng.randint(lo, hi)


def pick_interest_rate(rng: random.Random, loan_type: str) -> float:
    """Sample an annual interest rate (%) for the product."""
    lo, hi = INTEREST_RATES[loan_type]
    return round(rng.uniform(lo, hi), 2)


def pick_collateral(rng: random.Random, loan_type: str, loan_amount: int) -> int:
    """Sample collateral value (USD); secured products carry more collateral."""
    if loan_type in ("House", "Car"):
        return rng.randint(int(loan_amount * 0.5), int(loan_amount * 1.2))
    return rng.randint(0, loan_amount)


def annual_debt_service(loan_amount: int, interest_rate_pct: float, term_years: int) -> float:
    """Amortised annual payment using the standard annuity formula."""
    r = interest_rate_pct / 100.0
    if r <= 0:
        return loan_amount / term_years
    factor = (r * (1 + r) ** term_years) / ((1 + r) ** term_years - 1)
    return loan_amount * factor
