"""Carga de configuración declarativa desde YAML hacia `ExperimentConfig`."""

import pathlib

import yaml

from .schema import ExperimentConfig


def load_config(path) -> ExperimentConfig:
    """Lee un YAML de configuración de experimento y lo valida contra
    `ExperimentConfig`. Un campo faltante o mal tipado falla aquí, en la
    carga, y no a medio experimento."""
    text = pathlib.Path(path).read_text()
    data = yaml.safe_load(text)
    return ExperimentConfig(**data)
