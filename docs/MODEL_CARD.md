# Model Card — Loan Risk & Approval Models

Following the spirit of Mitchell et al. (2019), this card describes the two
models produced by `model_training`.

## Model details

- **Owner:** project author (portfolio / demonstration project).
- **Models:**
  - `risk_pipeline.joblib` — RandomForest multiclass classifier predicting
    `RiskCategory` ∈ {Low Risk, Medium Risk, High Risk}.
  - `approval_pipeline.joblib` — RandomForest binary classifier predicting
    `LoanStatus` ∈ {Approved, Rejected}.
- **Architecture:** `sklearn` `Pipeline` = `ColumnTransformer`
  (StandardScaler on 9 numeric features + OneHotEncoder on 6 categorical
  features) → `RandomForestClassifier`.
- **Hyper-parameters:** see `config.yaml → training.model`
  (200 trees, max depth 12, min samples leaf 5, balanced class weights).
- **Inputs:** 15 raw application features (no engineered leakage columns).
- **Outputs:** a single predicted class per model.

## Intended use

- **Primary use:** demonstration of an end-to-end synthetic-data + ML workflow;
  backing the [Loan-Advisor](https://github.com/salarymakage/Loan-Advisor)
  Django demo for interactive what-if scoring.
- **Intended users:** the project author, interviewers, and developers
  integrating the saved pipelines into a demo backend.

## Out-of-scope / prohibited use

- **Not for real lending decisions.** The models are trained entirely on
  synthetic data encoding hand-tuned assumptions; they have no validity for
  real applicants and must never be used to approve, deny, or price actual
  credit.
- No fairness auditing has been performed; `Gender`, `Province` and other
  sensitive-adjacent attributes are present and could encode bias.

## Training data

- 40,000 synthetic applications from `data_generation` (see
  [`METHODOLOGY.md`](METHODOLOGY.md) and [`DATA_DICTIONARY.md`](DATA_DICTIONARY.md)).
- 80/20 stratified train/test split, seed 42.

## Evaluation

Metrics are computed on the held-out 20% test set and written to
`artifacts/metrics.json` on every run. Indicative results:

| Model | Accuracy | Macro F1 |
|-------|:--------:|:--------:|
| Risk (3-class) | ~0.90 | ~0.86 |
| Approval (binary) | ~0.78 | ~0.76 |

- The **risk** model scores high because its label is a deterministic
  scorecard.
- The **approval** model is capped below ~0.80 by the deliberate stochasticity
  in the approval rule — this is expected, not a defect.

## Ethical considerations & caveats

- Synthetic data can launder unexamined assumptions into a model that *looks*
  rigorous. The transparent scorecard in `risk_engine.py` is provided precisely
  so those assumptions are inspectable.
- Sensitive attributes are included for realism; any real deployment would
  require a fairness review and likely their removal.
- The probabilistic approval label means the approval model reflects a *policy*,
  not an objective truth.
