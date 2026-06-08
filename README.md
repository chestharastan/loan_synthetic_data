# Cambodian Loan Risk & Approval — Synthetic Data & ML Pipeline

An end-to-end, reproducible machine-learning project that **(1) generates a
realistic synthetic loan-application dataset for the Cambodian micro-lending
market and (2) trains two models** on it:

| Model | Task | Target | Type |
|-------|------|--------|------|
| **Risk** | Credit-risk grading | `RiskCategory` ∈ {Low, Medium, High} | Multiclass classification |
| **Approval** | Underwriting decision | `LoanStatus` ∈ {Approved, Rejected} | Binary classification |

The data generation and the model training live in **separate, independent
packages** so each layer can evolve on its own. Everything is driven by a
single [`config.yaml`](config.yaml) and is reproducible from one random seed.

> The trained pipelines are saved as `joblib` artifacts ready to be served from
> a backend such as the companion Django app
> [Loan-Advisor](https://github.com/salarymakage/Loan-Advisor).

---

## Architecture

```
config.yaml ──────────────┐  (single source of truth)
                          ▼
 ┌─────────────────────────────┐        ┌──────────────────────────────┐
 │     data_generation/        │  CSV   │      model_training/         │
 │  distributions → risk_engine│ ─────► │ preprocessing → trainer       │
 │        → generator → run    │        │   → evaluate → predict        │
 └─────────────────────────────┘        └──────────────────────────────┘
            │                                        │
            ▼                                        ▼
     data/loan_applications.csv          models/*.joblib + artifacts/metrics.json
```

The two packages communicate **only through the CSV contract** documented in
[`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md). The trainer knows nothing
about how the data was produced; the generator knows nothing about the models.

---

## Project layout

```
loan_synthetic_data/
├── config.yaml                 # all tunable parameters live here
├── requirements.txt
├── Makefile                    # `make data`, `make train`, `make all`
│
├── common/                     # shared config loader
│   └── config.py
│
├── data_generation/            # ── FOLDER 1: produce the dataset ──
│   ├── distributions.py        #   domain sampling primitives
│   ├── risk_engine.py          #   deterministic risk scorecard + approval rule
│   ├── generator.py            #   assembles records → DataFrame → CSV
│   └── run.py                  #   CLI entry point
│
├── model_training/             # ── FOLDER 2: train & serve the models ──
│   ├── preprocessing.py        #   shared scaling + one-hot encoding
│   ├── trainer.py              #   fit, evaluate, persist both models
│   ├── evaluate.py             #   metrics helpers
│   ├── predict.py              #   load pipelines & score new applications
│   └── train.py                #   CLI entry point
│
├── data/                       # generated CSVs (git-ignored)
├── models/                     # saved .joblib pipelines (git-ignored)
├── artifacts/                  # metrics.json and plots (git-ignored)
├── docs/                       # data dictionary, methodology, model card
└── notebooks/                  # original exploratory notebooks (legacy)
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt           # or: make install

# 2. Generate the synthetic dataset → data/loan_applications.csv
python -m data_generation.run             # or: make data

# 3. Train both models → models/*.joblib + artifacts/metrics.json
python -m model_training.train            # or: make train

# 4. Score a sample application
python -m model_training.predict
```

Run the whole pipeline in one shot:

```bash
make all
```

### Useful overrides

```bash
python -m data_generation.run --n 5000 --out data/sample.csv
python -m model_training.train --data data/sample.csv
```

---

## Using the trained models

```python
from common import load_config
from model_training import LoanScorer

scorer = LoanScorer(load_config())

application = {
    "Province": "Kandal", "RegionType": "Semi-Urban", "Age": 43,
    "Gender": "Female", "EmploymentType": "Salaried", "AnnualIncomeUSD": 9000,
    "CreditHistory": "Good", "ExistingDebtUSD": 0, "SavingsAssetsUSD": 6550,
    "LoanType": "House", "LoanAmountUSD": 10775, "LoanTermYears": 10,
    "AnnualInterestRatePct": 14.23, "CollateralUSD": 12000, "DTI": 0.18,
}

print(scorer.score(application))
# {'risk_category': 'Low Risk', 'loan_status': 'Approved'}
```

---

## Results (40,000 synthetic applications, hold-out test set)

| Model | Accuracy | Macro F1 |
|-------|:--------:|:--------:|
| Risk (3-class) | ~0.90 | ~0.86 |
| Approval (binary) | ~0.78 | ~0.76 |

The risk model is highly learnable because its label is a transparent
scorecard. The approval model is intentionally harder — approval is a
*probabilistic* decision with deliberate noise, which is realistic and prevents
the model from trivially memorising the rule. Full metrics are written to
`artifacts/metrics.json` after each training run.

See [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) for intended use and limitations,
and [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) for how the data is generated
and labelled.

---

## Design highlights

- **Separation of concerns** — generation and training are decoupled packages
  that share nothing but a documented CSV schema.
- **Config-driven & reproducible** — one `config.yaml`, one seed; re-running
  reproduces the dataset and models bit-for-bit.
- **Transparent labels** — risk is a documented points-based scorecard
  (`risk_engine.py`), so the "ground truth" is auditable, not a black box.
- **Production-shaped** — `sklearn` `Pipeline`s bundle preprocessing with the
  model, so the saved `.joblib` consumes raw application dicts directly with no
  feature-engineering drift between training and serving.
- **Tested end-to-end** — `make all` runs generation → training → scoring.

---

## Legacy notebooks

The original exploratory notebooks are preserved under `notebooks/` for
reference:

- `Version1_legacy.ipynb` — early US-style dataset (33 features).
- `Version2_legacy.ipynb` — Cambodian dataset that this package productionises.

They are **not** part of the supported pipeline; use the packages above.
