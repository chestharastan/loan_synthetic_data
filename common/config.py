"""Typed access to ``config.yaml``.

A thin wrapper around the YAML file so that the rest of the codebase reads
configuration through attribute access (``cfg.generation["n_samples"]``) and
resolves paths relative to the project root, regardless of the working
directory the scripts are launched from.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Project root is the directory that contains this ``common`` package's parent.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@dataclass(frozen=True)
class Config:
    """In-memory view of ``config.yaml`` with convenience accessors."""

    raw: dict[str, Any]

    @property
    def project(self) -> dict[str, Any]:
        return self.raw["project"]

    @property
    def generation(self) -> dict[str, Any]:
        return self.raw["generation"]

    @property
    def training(self) -> dict[str, Any]:
        return self.raw["training"]

    @property
    def seed(self) -> int:
        return int(self.project["random_seed"])

    def path(self, relative: str) -> Path:
        """Resolve a config-declared path against the project root."""
        return PROJECT_ROOT / relative


def load_config(path: str | Path | None = None) -> Config:
    """Load and validate the project configuration."""
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config(raw=raw)
