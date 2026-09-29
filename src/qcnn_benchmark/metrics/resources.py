"""Métricas de recursos: qubits y parámetros entrenables de cada modelo, y
presupuestos de shots evaluados en E1 y E2. Los valores se leen de
`models/` y `execution/` en lugar de repetirse aquí, para que no se
desincronicen si esos módulos cambian.

Todo aplica igual a las QCNN y a sus análogas clásicas, salvo `n_wires`,
porque las análogas no tienen circuito."""

import pandas as pd

from qcnn_benchmark.execution.registry import get_circuit_spec
from qcnn_benchmark.execution.shots import n_trainable_params
from qcnn_benchmark.models import (
    cnn_gong_analoga,
    cnn_hur_analoga,
    cnn_wei_analoga,
    qcnn_cong,
    qcnn_gong,
    qcnn_hur,
    qcnn_wei,
)

_QUANTUM_MODULES = {"hur": qcnn_hur, "cong": qcnn_cong, "gong": qcnn_gong, "wei": qcnn_wei}
_CLASSICAL_MODULES = {
    "cnn_hur_analoga": cnn_hur_analoga,
    "cnn_gong_analoga": cnn_gong_analoga,
    "cnn_wei_analoga": cnn_wei_analoga,
}
_ALL_MODULES = {**_QUANTUM_MODULES, **_CLASSICAL_MODULES}


def _module(model_name):
    if model_name not in _ALL_MODULES:
        raise KeyError(f"'{model_name}' no está en el registro de recursos ({sorted(_ALL_MODULES)}).")
    return _ALL_MODULES[model_name]


def n_wires(model_name):
    """Qubits simulados por `model_name`, o `None` si es una análoga
    clásica. Wei no está en el registro genérico de circuitos
    (`execution/registry.py`), así que se toma de
    `qcnn_wei.N_WORK_QUBITS`."""
    if model_name not in _QUANTUM_MODULES:
        return None
    if model_name == "wei":
        return qcnn_wei.N_WORK_QUBITS
    return get_circuit_spec(model_name).n_wires


def total_params(model_name):
    """Parámetros totales del modelo (`TOTAL_PARAMS` de su módulo)."""
    return _module(model_name).TOTAL_PARAMS


def trainable_params(model_name, regime="analytic"):
    """Parámetros efectivamente entrenados bajo `regime`.

    Con "analytic" (E0, E0A y E0B, sin shots) son todos los `TOTAL_PARAMS`,
    incluido Wei completo (filtro LCU y Hamiltoniano de lectura).
    Con "shots" o "noise" (E1 y E2) el resultado es el mismo salvo para Wei,
    que fija el filtro LCU (`beta`) y solo entrena el Hamiltoniano. La
    razón está en `execution.shots.make_wei_predict_proba`.
    """
    if regime == "analytic" or model_name not in _QUANTUM_MODULES:
        return total_params(model_name)
    if regime in ("shots", "noise"):
        return n_trainable_params(model_name)
    raise ValueError(f"regime desconocido: {regime!r} (usa 'analytic', 'shots' o 'noise')")


def resource_summary(model_names, regime="analytic"):
    """Tabla con qubits, parámetros totales y parámetros entrenados bajo
    `regime` para cada modelo de `model_names`."""
    rows = [
        {
            "model": name,
            "n_wires": n_wires(name),
            "total_params": total_params(name),
            "trainable_params": trainable_params(name, regime=regime),
        }
        for name in model_names
    ]
    return pd.DataFrame(rows)


def shots_budget_summary(df):
    """Valores únicos y ordenados de `n_shots` en un CSV de E1 o E2 cargado
    con pandas."""
    return sorted(df["n_shots"].unique().tolist())
