"""CNN clásica análoga al QCNN de Hur et al. (`qcnn_hur`, Ansatz 8 / U_6,
36 parámetros entrenables).

Arquitectura compartida en `_cnn_common` (Conv1d -> ReLU -> MaxPool1d ->
Conv1d -> ReLU -> MaxPool1d -> Linear -> sigmoide), con la dimensión de
entrada elegida para igualar aproximadamente ese conteo de parámetros. Es
el criterio de Gong et al. (2024), Tabla 4, que ajustan la dimensión de PCA
para que la CNN tenga un tamaño cercano al de la QCNN comparada (PCA-8 da
26 parámetros contra 24, y PCA-16 da 34 contra 36).

Con `N_FEATURE=2` canales, igual que la CNN de referencia de Hur et al., y
PCA-36 la CNN tiene 35 parámetros frente a los 36 de
`qcnn_hur.TOTAL_PARAMS`. La diferencia de 1 queda dentro del margen de las
comparaciones de Gong, que difieren en 2.

Espera datos con la representación PCA densa de
qcnn_benchmark.representations.pca con n_components=INPUT_SIZE.
"""

from qcnn_benchmark.models import _cnn_common

N_FEATURE = 2
INPUT_SIZE = 36
TOTAL_PARAMS = _cnn_common.total_params(INPUT_SIZE, N_FEATURE)
assert TOTAL_PARAMS == 35  # qcnn_hur.TOTAL_PARAMS es 36

# Kaiming en vez de normal_init, ver _cnn_common.make_kaiming_init
init = _cnn_common.make_kaiming_init(INPUT_SIZE, N_FEATURE)


def predict_proba(params, x):
    return _cnn_common.forward_proba(params, x, n_feature=N_FEATURE)
