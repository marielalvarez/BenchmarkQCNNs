"""CNN clásica análoga al QCNN de Gong et al. (`qcnn_gong`, Circuit 6 +
codificación tree-hybrid, 36 parámetros entrenables).

Arquitectura compartida en `_cnn_common` (Conv1d -> ReLU -> MaxPool1d ->
Conv1d -> ReLU -> MaxPool1d -> Linear -> sigmoide), con la dimensión de
entrada elegida para igualar aproximadamente ese conteo de parámetros. Es
el mismo criterio que usan Gong et al. (2024) en su Tabla 4, donde comparan
su QCNN contra una CNN de tamaño cercano (PCA-8 da una CNN de 26 contra
QCNN-Circuit5 de 24, y PCA-16 da 34 contra QCNN-Circuit6 de 36).

Con `N_FEATURE=2` canales y PCA-36 la CNN tiene 35 parámetros, frente a
los 36 de `qcnn_gong.TOTAL_PARAMS`. La configuración es idéntica a la de
`cnn_hur_analoga` a propósito, porque el Ansatz 8 de Hur y el Circuit 6 de
Gong se construyen ambos sobre `unitary.U_6` y tienen el mismo conteo.

Espera datos con la representación PCA densa de
qcnn_benchmark.representations.pca con n_components=INPUT_SIZE. Esto es
independiente de la capacidad de la codificación tree-hybrid de la QCNN de
Gong, que el análogo clásico no usa.
"""

from qcnn_benchmark.models import _cnn_common

N_FEATURE = 2
INPUT_SIZE = 36
TOTAL_PARAMS = _cnn_common.total_params(INPUT_SIZE, N_FEATURE)
assert TOTAL_PARAMS == 35  # qcnn_gong.TOTAL_PARAMS es 36

# Kaiming en vez de normal_init, ver _cnn_common.make_kaiming_init
init = _cnn_common.make_kaiming_init(INPUT_SIZE, N_FEATURE)


def predict_proba(params, x):
    return _cnn_common.forward_proba(params, x, n_feature=N_FEATURE)
