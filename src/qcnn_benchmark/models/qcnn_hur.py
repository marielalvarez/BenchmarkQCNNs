"""Adaptador de Ansatz 8 de Hur, Kim & Park (2022), "Quantum convolutional
neural network for classical data classification".

Reutiliza sin modificar el código de los autores (Apache-2.0,
https://github.com/takh04/QCNN), que se espera en external/hur_qcnn/:

- unitary.U_6, el ansatz convolucional de 2 qubits ("Ansatz 8" del paper).
- unitary.Pooling_ansatz1, pooling con CRZ + PauliX + CRX.
- QCNN_circuit.QCNN, el qnode completo (embedding, estructura 8->4->2->1
  con parámetros compartidos y lectura), llamado con U='U_6', U_params=10
  y embedding_type='Angle-compact'. Esta es la codificación densa, con
  AngleEmbedding en X para las primeras 8 componentes y en Y para las 8
  siguientes.

El "Ansatz 8" se identificó cruzando la Fig. 2(h) del paper con la Tabla I
(fila 8, 36 parámetros, columna Dense+PCA con 98.7 ± 0.1%). Coincide gate
por gate con la función U_6 de unitary.py.

Espera datos con la representación PCA densa de
qcnn_benchmark.representations.pca (16 componentes, escaladas a [0, π]).
"""

import pathlib
import sys

_MODULE_DIR = pathlib.Path(__file__).resolve()
PROJECT_ROOT = _MODULE_DIR
while not (PROJECT_ROOT / "external").exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent
HUR_QCNN_DIR = PROJECT_ROOT / "external" / "hur_qcnn" / "QCNN"
assert HUR_QCNN_DIR.exists(), f"No se encontró {HUR_QCNN_DIR}"

if str(HUR_QCNN_DIR) not in sys.path:
    sys.path.insert(0, str(HUR_QCNN_DIR))

import QCNN_circuit as _hur_circuit  # noqa: E402  (vendorizado, Apache-2.0, takh04/QCNN)
import unitary as _hur_unitary  # noqa: E402  (vendorizado, Apache-2.0, takh04/QCNN)

assert hasattr(_hur_unitary, "U_6")
assert hasattr(_hur_unitary, "Pooling_ansatz1")

U_PARAMS = 10  # parámetros de U_6 por aplicación
N_POOL_PARAMS = 2  # parámetros de Pooling_ansatz1 por aplicación
TOTAL_PARAMS = 3 * U_PARAMS + 3 * N_POOL_PARAMS  # 36, Tabla I del paper, fila Ansatz 8


def forward_probs(x, params):
    """qml.probs(wires=4) del qnode QCNN_circuit.QCNN sin modificar, con
    U='U_6' (Ansatz 8) y codificación densa ('Angle-compact')."""
    return _hur_circuit.QCNN(
        x, params, U="U_6", U_params=U_PARAMS, embedding_type="Angle-compact", cost_fn="cross_entropy"
    )


def predict_proba(params, x):
    """Probabilidad de clase positiva (índice 1 de qml.probs(wires=4)). La
    convención se verificó contra Benchmarking.accuracy_test del
    repositorio de los autores."""
    return forward_probs(x, params)[1]
