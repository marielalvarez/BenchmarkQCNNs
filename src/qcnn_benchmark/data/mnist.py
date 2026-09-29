"""Carga de MNIST y Fashion-MNIST vía sklearn.datasets.fetch_openml.

Se usa fetch_openml en vez de tf.keras.datasets para no depender de
tensorflow, cuya versión 2.20.0 tiene un conflicto de protobuf que rompe
su import. fetch_openml entrega las mismas 70,000 imágenes y etiquetas de
cada dataset.
"""

import numpy as np
from sklearn.datasets import fetch_openml


def load_mnist_pool(normalize=True):
    """Todas las 70,000 imágenes de MNIST (train+test oficiales combinados).

    normalize=True escala a [0, 1] dividiendo por 255 (usado por la
    representación PCA de Hur et al.). normalize=False deja los valores
    crudos en [0, 255] (usado por la representación de amplitud de Wei et
    al., que normaliza en norma L2 más adelante y por lo tanto es invariante
    a este escalado).
    """
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    x_all = mnist.data.astype(np.float64)
    if normalize:
        x_all = x_all / 255.0
    y_all = mnist.target.astype(int)
    return x_all, y_all


def load_fashion_mnist_pool(normalize=False):
    """Todas las 70,000 imágenes de Fashion-MNIST (train+test combinados)."""
    fashion = fetch_openml("Fashion-MNIST", version=1, as_frame=False, parser="auto")
    x_all = fashion.data.astype(np.float64)
    if normalize:
        x_all = x_all / 255.0
    y_all = fashion.target.astype(int)
    return x_all, y_all
