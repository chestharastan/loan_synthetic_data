# Methodology — How the Synthetic Data Is Built and Labelled

This document explains the *why* behind the generator so a reviewer can judge
whether the synthetic data is realistic and whether the learning task is sound.

## 1. Goals

1. Produce a loan-application dataset that reflects the **Cambodian
   micro-lending market** (provinces, products, USD amounts, local interest
   rates) without using any real customer data.
2. Attach **transparent, rule-based labels** so the ground truth is auditable
   and the ML task is well-posed.
3. Be **fully reproducible** from a single seed.

## 2. Generation pipeline

Each application is built in a fixed causal order so that downstream variables
depend on upstream ones — this is what creates realistic correlations rather
than independent noise.

```
Province ─► RegionType ─► EmploymentType ─► AnnualIncome
                                   │
   Age ────────────────────────────┘
                                   ▼
                LoanType ─► LoanAmount ─► Term, InterestRate ─► DTI
                                   ▼
                  RiskScore ─► RiskCategory ─► LoanStatus
```

Key conditional relationships (all in `data_generation/distributions.py`):

- **Region drives employment.** Rural areas skew toward farming; urban areas
  skew toward salaried/self-employed work.
- **Income depends on region, employment and age.** Prime working age (25–45)
  gets a +20% ceiling; older applicants a −10% ceiling.
- **Loan product depends on segment.** Farmers favour agriculture/moto loans;
  salaried workers access housing and car loans.
- **Loan amount is capped at 5× income**, mirroring sensible exposure limits.
- **Terms and rates are product-specific** (e.g. housing 9.6–18% over 5–25
  years; moto 24–36% over 1–3 years), reflecting real Cambodian MFI ranges.
- **DTI** uses the standard amortising-annuity formula.

## 3. Labelling

### Risk (`risk_engine.score_risk`)

A transparent **points-based scorecard** in `[0, 100]`. Each factor contributes
risk points; mitigants subtract them:

| Factor | Max points | Rationale |
|--------|:----------:|-----------|
| Employment type | 25 | Income stability |
| Loan-to-income ratio | 20 | Leverage |
| Loan purpose | 15 | Product loss history |
| Debt-to-income (DTI) | 15 | Repayment capacity |
| Credit history | 12 | Past behaviour |
| Region | 10 | Collection/operational risk |

Mitigants: collateral ≥ loan, savings ≥ income, rural agriculture loans, and
moto loans for salaried workers. Aggravators: high existing debt, very young or
old applicants. The score is bucketed into Low (≤35), Medium (36–65), High
(>65).

Because the label is a deterministic function of the features, the **risk model
is highly learnable (~90% accuracy)** — which is the expected, healthy
behaviour for a rules-derived target.

### Approval (`risk_engine.decide_approval`)

Approval is intentionally **probabilistic**, not deterministic:

1. A base approval probability is set by risk band (Low 85%, Medium 45%, High
   15%).
2. It is nudged by collateral, credit history, employment and product.
3. A Bernoulli draw decides the final outcome.

This injects irreducible noise, so the approval model **cannot perfectly
recover the rule (~78% accuracy)**. That is deliberate: it makes the binary
task realistic and avoids a degenerate "the model memorised the rule" result.

## 4. Reproducibility

- A single `random.Random(seed)` instance drives **every** draw; the seed lives
  in `config.yaml → project.random_seed`.
- Re-running `python -m data_generation.run` regenerates the identical dataset.
- `train_test_split` and the RandomForest are seeded with the same value, so
  training is reproducible too.

## 5. Limitations

- The data is **synthetic** and encodes the author's assumptions about the
  market; it is not validated against real loan books.
- Distributions are hand-tuned, not fitted to survey data.
- See [`MODEL_CARD.md`](MODEL_CARD.md) for downstream-use limitations.
