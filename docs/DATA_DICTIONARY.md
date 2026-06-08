# Data Dictionary — `loan_applications.csv`

The dataset produced by `data_generation` is the single contract between the
two packages. Each row is one loan application. Columns are emitted in the
order below (see `COLUMN_ORDER` in `data_generation/generator.py`).

## Identifier

| Column | Type | Description |
|--------|------|-------------|
| `ApplicationID` | int | Sequential 1..N application identifier. Not a model feature. |

## Features (model inputs)

| Column | Type | Domain / Range | Description |
|--------|------|----------------|-------------|
| `Province` | categorical | Phnom Penh, Kandal, Siem Reap, Svay Rieng, Kampong Speu, Kampong Thom, Battambang, Others | Applicant's province. |
| `RegionType` | categorical | Urban, Semi-Urban, Rural | Derived from province via `region_map`. |
| `Age` | int | 18–65 | Applicant age in years. |
| `Gender` | categorical | Male, Female | Applicant gender. |
| `EmploymentType` | categorical | Salaried, Self-Employed, Farmer, Student, Unemployed | Conditioned on region. |
| `AnnualIncomeUSD` | int | ≥ 0 | Annual income; depends on region, employment, age. |
| `CreditHistory` | categorical | Good, Fair, Poor, No History | Credit-bureau style grade. |
| `ExistingDebtUSD` | int | ≥ 0 | Outstanding debt (0 for ~65% of applicants). |
| `SavingsAssetsUSD` | int | ≥ 0 | Liquid savings / assets, up to ~2× income. |
| `LoanType` | categorical | House, Car, Moto, Personal, SME, Agriculture | Requested product; conditioned on region & employment. |
| `LoanAmountUSD` | int | product/region range, capped at 5× income | Requested principal. |
| `LoanTermYears` | int | product-specific (1–25) | Loan duration in years. |
| `AnnualInterestRatePct` | float | product-specific (9.6–36.0) | Nominal annual interest rate (%). |
| `CollateralUSD` | int | ≥ 0 | Pledged collateral value. Higher for House/Car. |
| `DTI` | float | 0–99 (capped) | Debt-to-income: annual amortised payment ÷ annual income. |

> `DTI` is computed with the standard annuity formula in
> `distributions.annual_debt_service`. When income is zero, it is capped at the
> finite sentinel `99.0` so the value stays model-safe.

## Targets (labels)

| Column | Type | Domain | Description |
|--------|------|--------|-------------|
| `RiskScore` | int | 0–100 | Raw points from the risk scorecard. Diagnostic only — **not** a model input. |
| `RiskCategory` | categorical | Low Risk (≤35), Medium Risk (36–65), High Risk (>65) | **Target** of the risk model. |
| `LoanStatus` | categorical | Approved, Rejected | **Target** of the approval model. |

## Important modelling notes

- `RiskScore` is **excluded** from the model feature set on purpose. It is the
  intermediate that defines `RiskCategory`; feeding it in would leak the label.
- `RiskCategory` is **not** used as a feature for the approval model either.
  Both models consume the same 15 raw application features (9 numeric + 6
  categorical) listed in `config.yaml → training.features`.
- The feature/label split is enforced in code, so the saved pipelines never see
  leakage columns.
