"""Synthetic loan-application data generation for the Cambodian lending context.

The package is intentionally separate from ``model_training`` so that the data
layer can evolve independently of the modelling layer.

Modules
-------
distributions : domain sampling primitives (demographics, income, loan terms)
risk_engine   : deterministic risk-scoring and approval rules
generator     : assembles a full application record and the final DataFrame
run           : command-line entry point (``python -m data_generation.run``)
"""

from .generator import LoanDataGenerator

__all__ = ["LoanDataGenerator"]
