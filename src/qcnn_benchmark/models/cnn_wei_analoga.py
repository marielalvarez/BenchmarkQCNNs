"""CNN clásica análoga al QCNN de Wei et al. (`qcnn_wei`, filtro LCU +
lectura hamiltoniana, 46 parámetros entrenables).

Arquitectura compartida en `_cnn_common` (Conv1d -> ReLU -> MaxPool1d ->
Conv1d -> ReLU -> MaxPool1d -> Linear -> sigmoide), con la dimensión de
entrada y el número de canales elegidos para igualar el conteo de
parámetros, siguiendo el criterio de Gong et al. (2024), Tabla 4.

Con `N_FEATURE=3` canales y PCA-20 la CNN tiene 46 parámetros, exactamente
los de `qcnn_wei.TOTAL_PARAMS`. `qcnn_wei` usa codificación de amplitud
directa sin PCA (1024 = 32x32 px), pero el análogo clásico sí reduce con
PCA. Gong et al. hacen lo mismo con su CNN de referencia sin importar la
codificación de la QCNN comparada.

Espera datos con la representación PCA densa de
qcnn_benchmark.representations.pca con n_components=INPUT_SIZE.
"""

from qcnn_benchmark.models import _cnn_common

N_FEATURE = 3
INPUT_SIZE = 20
TOTAL_PARAMS = _cnn_common.total_params(INPUT_SIZE, N_FEATURE)
assert TOTAL_PARAMS == 46  # igual a qcnn_wei.TOTAL_PARAMS

# Kaiming en vez de normal_init, ver _cnn_common.make_kaiming_init
init = _cnn_common.make_kaiming_init(INPUT_SIZE, N_FEATURE)


def predict_proba(params, x):
    return _cnn_common.forward_proba(params, x, n_feature=N_FEATURE)
