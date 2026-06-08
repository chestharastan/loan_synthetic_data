"""Feature preprocessing shared by the risk and approval models.

A single factory builds the ``ColumnTransformer`` so both pipelines apply the
exact same scaling and encoding, driven by the feature lists in ``config.yaml``.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import Config


def build_preprocessor(config: Config) -> ColumnTransformer:
    """Create the standard scaler + one-hot encoder column transformer."""
    features = config.training["features"]
    numeric = features["numeric"]
    categorical = features["categorical"]

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )


def feature_columns(config: Config) -> list[str]:
    """Return the full ordered list of model input columns."""
    features = config.training["features"]
    return list(features["numeric"]) + list(features["categorical"])
