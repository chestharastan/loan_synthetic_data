"""Model training, evaluation and inference for loan risk & approval.

Kept separate from ``data_generation`` so the modelling layer only depends on
the CSV contract, not on how the data was produced.

Modules
-------
preprocessing : shared ColumnTransformer (scaling + one-hot encoding)
trainer       : fit, evaluate and persist the two RandomForest pipelines
evaluate      : metric helpers (classification report, confusion matrix)
predict       : load saved pipelines and score new applications
train         : command-line entry point (``python -m model_training.train``)
"""

from .trainer import ModelTrainer
from .predict import LoanScorer

__all__ = ["ModelTrainer", "LoanScorer"]
