"""Adaptador de la arquitectura QCNN genérica de Cong, Choi & Lukin (2019),
"Quantum convolutional neural networks" (Nat. Phys. 15, 1273-1278).

Cong et al. no proponen una tarea de clasificación de imágenes clásicas ni
un esquema de codificación de datos clásicos: su QCNN opera directamente
sobre estados cuánticos de entrada (reconocimiento de fases topológicas
protegidas por simetría, optimización de códigos de corrección de
errores). Lo que se toma de ellos, como arquitectura de referencia
genérica para E0B, es el patrón estructural de la Fig. 1b. Una capa
convolucional aplica un único unitario de 2 qubits U_i de forma invariante
a traslación, y una capa de pooling mide una fracción de los qubits y
aplica una rotación condicionada V_j a los restantes. Esto se repite hasta
dejar un solo qubit de lectura.

E0B compara esta QCNN contra Hur, Wei y Gong para ver si las aportaciones
de cada paper (codificación densa, filtro LCU con hamiltoniano,
codificación tree-hybrid) mejoran algo frente a una QCNN jerárquica con el
unitario de 2 qubits más expresivo y la codificación más neutra posibles.

Reutiliza sin modificar el código de Hur et al. (Apache-2.0,
https://github.com/takh04/QCNN), igual que `qcnn_hur.py`:

- unitary.U_SU4, el unitario general de 2 qubits (SU(4) completo con la
  descomposición U3+CNOT de Vatan y Williams, 15 parámetros). Puede
  representar cualquier operación de 2 qubits, así que no favorece a
  ninguna tarea ni codificación.
- unitary.Pooling_ansatz1, el mismo pooling parametrizado (CRZ + X + CRX,
  2 parámetros) de `qcnn_hur.py`. Es una versión simulable y
  diferenciable del pooling de la Fig. 1b de Cong et al.
- embedding.data_embedding(..., embedding_type='Angle'), un
  AngleEmbedding estándar con una feature por qubit y rotación Y. A
  diferencia de la codificación densa de Hur o la tree-hybrid de Gong, no
  forma parte de la aportación de ningún paper.
- QCNN_circuit.QCNN_structure, la topología jerárquica 8->4->2->1 que se
  usa tal cual a través de QCNN_circuit.QCNN con U='U_SU4'.

Espera datos con la representación PCA densa de
qcnn_benchmark.representations.pca con n_components=8 (una componente por
qubit), escalada a [0, π].
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

assert hasattr(_hur_unitary, "U_SU4")

U_PARAMS = 15  # parámetros de U_SU4 por aplicación (SU(4) completo)
N_POOL_PARAMS = 2  # parámetros de Pooling_ansatz1 por aplicación
TOTAL_PARAMS = 3 * U_PARAMS + 3 * N_POOL_PARAMS  # 51


def forward_probs(x, params):
    """qml.probs(wires=4) del qnode QCNN_circuit.QCNN sin modificar, con
    U='U_SU4' y AngleEmbedding estándar."""
    return _hur_circuit.QCNN(x, params, U="U_SU4", U_params=U_PARAMS, embedding_type="Angle", cost_fn="cross_entropy")


def predict_proba(params, x):
    """Probabilidad de clase positiva (índice 1 de qml.probs(wires=4)), con
    la misma convención que qcnn_hur.predict_proba."""
    return forward_probs(x, params)[1]
