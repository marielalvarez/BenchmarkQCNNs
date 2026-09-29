"""Esquema Pydantic, carga de YAML y hashing canónico de configuración."""

from .loader import load_config
from .schema import DatasetSpec, ExperimentConfig, StatisticalCriterion, TrainingProtocol

__all__ = [
    "DatasetSpec",
    "TrainingProtocol",
    "StatisticalCriterion",
    "ExperimentConfig",
    "load_config",
]
