"""Motor de ejecución de E1 y E2: shots finitos y transpilación a un mapa de
acoplamiento fijo. `qcnn_benchmark.noise` combina ambas piezas con canales
de ruido para E2.
"""

from .registry import CIRCUIT_SPECS, CircuitSpec, get_circuit_spec
from .shots import make_shots_predict_proba
from .transpile import COUPLING_MAP_16Q, grid_coupling_map, transpile_qnode

__all__ = [
    "CircuitSpec",
    "CIRCUIT_SPECS",
    "get_circuit_spec",
    "make_shots_predict_proba",
    "transpile_qnode",
    "grid_coupling_map",
    "COUPLING_MAP_16Q",
]
