# Synthetic Loan Data Generation and Risk/Approval Modelling — Project Report

**A research-style write-up of the synthetic loan-application project.**

---

## Abstract

Access to real consumer-lending data is restricted by privacy law and
commercial confidentiality, which makes it hard to build and demonstrate credit
models openly. This project addresses that gap by **generating realistic
synthetic loan-application data** and training machine-learning models on it to
predict two outcomes: an applicant's **credit-risk grade** and the lender's
**approval decision**. The work is organised into two iterations — *Version 1*,
a broad US-style dataset with 33 features, and *Version 2*, a focused dataset
modelled on the **Cambodian micro-lending market** with 15 features. Both use a
transparent, rule-based labelling scheme so that the "ground truth" is auditable
rather than a black box. On held-out test data the risk model reaches ~91%
accuracy and the approval model ~78%, with the gap being a *deliberate*
consequence of injecting realistic stochasticity into the approval rule. The
deliverable is a reproducible, config-driven pipeline that separates data
generation from model training, with trained pipelines exportable to a web
backend.

---

## 1. Introduction

### 1.1 Problem statement

Lenders decide (a) **how risky** a borrower is, and (b) **whether to approve**
the loan. Building ML models for these tasks normally requires historical loan
books, which are private. Without data, one cannot prototype, teach, or
interview around the problem. **Can we instead synthesise data that is realistic
enough to support meaningful modelling, while keeping the labelling logic fully
transparent?**

### 1.2 Objectives

1. Generate a synthetic loan-application dataset that encodes plausible
   socio-economic correlations (income, region, employment, loan products).
2. Attach rule-based labels for **risk grade** and **approval status**.
3. Train and evaluate models that recover those labels.
4. Make the whole thing **reproducible** and **production-shaped** (saved
   pipelines that can be served from an app).

### 1.3 Contributions

- A documented, seed-reproducible **synthetic data generator** for the
  Cambodian lending context.
- A transparent **points-based risk scorecard** and a **probabilistic approval
  rule** that together define the learning targets.
- Two trained `scikit-learn` pipelines plus an evaluation harness.
- A clean software design that **decouples data generation from training**, the
  two communicating only through a documented CSV schema.

---

## 2. Background and motivation

### 2.1 Why synthetic data

Synthetic data sidesteps privacy and licensing constraints and lets us *control*
the data-generating process. The trade-off is that any model trained on it only
learns the assumptions baked into the generator. We mitigate this by keeping
those assumptions **explicit and inspectable** (`risk_engine.py`,
`distributions.py`) rather than hidden.

### 2.2 Why the Cambodian context (Version 2)

Version 1 used generic US-style ranges. Version 2 was re-grounded in the
Cambodian micro-finance market because it makes the synthetic ranges concrete
and defensible: USD-denominated loans, province-level geography, product types
common to Cambodian MFIs (moto, agriculture, SME), and the high nominal interest
rates typical of the sector (e.g. 24–36% on moto loans). This makes the dataset
feel like a real domain rather than abstract noise.

---

## 3. The two versions

The project evolved through two iterations. Both are preserved in `notebooks/`.

| Aspect | **Version 1 (legacy)** | **Version 2 (productionised)** |
|--------|------------------------|--------------------------------|
| Market | US-style, generic | Cambodian micro-lending |
| Feature count | 33 features | 15 features |
| Targets | `LoanApproved` (classification), `RiskScore` (regression) | `RiskCategory` (3-class), `LoanStatus` (binary) |
| Risk label | Weighted sub-scores across many factors | Points scorecard in [0, 100] → 3 bands |
| Approval label | Deterministic threshold on an aggregate score | Probabilistic, conditioned on risk band |
| Models | Logistic Regression, Random Forest, Gradient Boosting | Random Forest pipelines (one per target) |
| Persistence | None (notebook only) | Saved `joblib` pipelines + metrics |
| Status | Exploratory reference | The supported, packaged pipeline |

### 3.1 Version 1 — broad feature exploration

Version 1 generated **40,000** records with **33 features** spanning
demographics (age, education, marital status, dependents), credit behaviour
(credit score, utilisation, inquiries, payment history, bankruptcies), and
balance-sheet items (assets, liabilities, net worth, savings, checking). It
derived an `InterestRate` from credit score / amount / term, computed a
`TotalDebtToIncomeRatio`, then applied:

- a **deterministic approval rule** — a weighted sum of credit, income, DTI,
  loan size, interest, bankruptcy and employment factors, thresholded to a
  binary `LoanApproved`;
- a **risk score** — a second weighted aggregation bucketed into Low / Medium /
  High risk.

It trained Logistic Regression, Random Forest and Gradient Boosting models,
reporting ~91% approval accuracy and an R² of ~0.76 on the risk-score
regression. Its weakness was **complexity**: 33 features, several of them only
weakly motivated, and no model persistence.

### 3.2 Version 2 — focused, domain-grounded, productionised

Version 2 reduced the feature set to the **15 most decision-relevant variables**
and re-grounded them in Cambodia. It introduced the **points-based scorecard**
for risk and a **probabilistic approval** rule, trained one Random Forest
pipeline per target, and **saved the pipelines** for serving. This version is
the one rebuilt into the package described below; Version 1 remains as a
reference notebook.

---

## 4. Methodology

### 4.1 Data-generating process (Version 2)

Each application is constructed in a fixed **causal order**, so downstream
variables depend on upstream ones, producing realistic correlations rather than
independent noise:

```
Province → RegionType → EmploymentType → AnnualIncome
                                │
   Age ─────────────────────────┘
                                ▼
             LoanType → LoanAmount → Term, InterestRate → DTI
                                ▼
               RiskScore → RiskCategory → LoanStatus
```

Key conditional rules (in `data_generation/distributions.py`):

- **Region drives employment:** rural areas skew to farming; urban to
  salaried/self-employed.
- **Income depends on region, employment and age:** prime age (25–45) gets a
  higher ceiling; older applicants a lower one.
- **Product depends on segment:** farmers favour agriculture/moto; salaried
  workers access housing/car loans.
- **Loan amount capped at 5× income** for plausible exposure.
- **Term and rate are product-specific** (housing 9.6–18% over 5–25 yrs; moto
  24–36% over 1–3 yrs).
- **DTI** uses the standard amortising-annuity formula.

### 4.2 Labelling

**Risk** (`risk_engine.score_risk`) is a transparent scorecard in `[0, 100]`:

| Factor | Max points |
|--------|:----------:|
| Employment type | 25 |
| Loan-to-income ratio | 20 |
| Loan purpose | 15 |
| Debt-to-income (DTI) | 15 |
| Credit history | 12 |
| Region | 10 |

Mitigants (collateral ≥ loan, savings ≥ income, rural agriculture, salaried moto
loans) subtract points; aggravators (heavy existing debt, very young/old
applicants) add them. The score is bucketed: **Low ≤ 35**, **Medium 36–65**,
**High > 65**.

**Approval** (`risk_engine.decide_approval`) is intentionally **probabilistic**:
a base probability by risk band (Low 85%, Medium 45%, High 15%) is nudged by
collateral, credit history, employment and product, then a Bernoulli draw
decides the outcome. This injects irreducible noise so the task is non-trivial.

### 4.3 Modelling

Both targets are learned by independent **Random Forest** classifiers wrapped in
a `scikit-learn` `Pipeline`:

```
ColumnTransformer( StandardScaler(numeric) + OneHotEncoder(categorical) )
        → RandomForestClassifier(200 trees, max_depth 12, balanced classes)
```

Bundling preprocessing inside the pipeline guarantees **train/serve
consistency** — the saved `.joblib` consumes raw application dictionaries with
no separate feature-engineering step that could drift.

### 4.4 Experimental setup

- 40,000 synthetic applications.
- 80/20 **stratified** train/test split, seed 42.
- Metrics: accuracy, macro/weighted F1, confusion matrix, per-class report
  (written to `artifacts/metrics.json`).

### 4.5 Avoiding label leakage

`RiskScore` (the intermediate that defines `RiskCategory`) and `RiskCategory`
itself are **excluded** from the feature set. Both models consume the same 15
raw features, enforced in code, so the saved pipelines never see leakage
columns.

---

## 5. Results

On the held-out test set (8,000 applications):

| Model | Accuracy | Macro F1 |
|-------|:--------:|:--------:|
| **Risk** (Low / Medium / High) | ~0.91 | ~0.89 |
| **Approval** (Approved / Rejected) | ~0.78 | ~0.76 |

Representative risk confusion matrix (rows = true, cols = predicted; order Low,
Medium, High):

```
[[4061  278    0]
 [ 171 2624  192]
 [   0   56  618]]
```

**Interpretation.**

- The **risk** model is highly accurate because its label is a deterministic
  function of the features; a tree ensemble recovers a scorecard well. Errors
  concentrate on adjacent bands (Low↔Medium, Medium↔High), never skipping a band
  — exactly the failure mode one would expect and tolerate.
- The **approval** model plateaus near 0.78 **by design**: the approval label
  contains a Bernoulli coin-flip, so part of the outcome is irreducible. A model
  that scored ~0.99 here would signal a leak or a broken (deterministic) rule.

---

## 6. Discussion

### 6.1 What the project demonstrates

- A complete, reproducible ML workflow from data synthesis to a serveable model.
- Disciplined **separation of concerns**: generation and training are
  independent packages sharing only a documented CSV contract.
- **Honest evaluation**: the approval ceiling is explained, not hidden.

### 6.2 Engineering choices worth noting

- **Config-driven** (`config.yaml`): sample size, distributions, features and
  hyper-parameters are data, not code.
- **Single-seed reproducibility**: one RNG drives every draw.
- **Auditable ground truth**: the scorecard is readable, so reviewers can check
  the data is sensible before trusting any metric.

### 6.3 Limitations

- The data is **synthetic** and encodes hand-tuned assumptions; it is not
  validated against a real loan book.
- Distributions are designed, not fitted to survey data.
- Sensitive-adjacent attributes (`Gender`, `Province`) are present for realism;
  **no fairness audit** has been done. The models must not be used for real
  lending decisions (see `MODEL_CARD.md`).

### 6.4 Future work

- Calibrate distributions against published Cambodian MFI statistics.
- Add probability-calibrated outputs and decision thresholds for the approval
  model.
- Add a fairness analysis across gender/region.
- Gradient-boosting and logistic baselines for comparison (as in Version 1).
- Unit tests for the scorecard and samplers.

---

## 7. Conclusion

This project shows that a carefully designed **synthetic** dataset can support a
credible, end-to-end credit-modelling workflow without any real customer data.
By grounding Version 2 in the Cambodian micro-lending market and keeping the
labelling logic transparent, the pipeline produces models whose strengths (a
highly learnable risk grade) and limits (an intentionally noisy approval task)
are both **explainable**. The result is a reproducible, production-shaped
codebase suitable for demonstration, teaching, and integration into a web
application.

---

## Appendix A — Reproducing the results

```bash
pip install -r requirements.txt
python -m data_generation.run        # → data/loan_applications.csv
python -m model_training.train       # → models/*.joblib, artifacts/metrics.json
python -m model_training.predict     # score a sample application
# or simply:
make all
```

## Appendix B — Related documents

- `docs/METHODOLOGY.md` — deeper detail on generation and labelling.
- `docs/DATA_DICTIONARY.md` — every column, type and range.
- `docs/MODEL_CARD.md` — intended use, limitations, ethics.
- `notebooks/Version1_legacy.ipynb`, `notebooks/Version2_legacy.ipynb` — the
  original exploratory work.
