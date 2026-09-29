import numpy as np

from qcnn_benchmark.data.sampling import stratified_split
from qcnn_benchmark.representations.amplitude import amplitude_representation
from qcnn_benchmark.representations.pca import dense_pca_representation


def _fake_image_pool(n_per_class=40, img_size=8, seed=1):
    rng = np.random.default_rng(seed)
    n = n_per_class * 2
    x = rng.uniform(0, 1, size=(n, img_size * img_size))
    y = np.array([0] * n_per_class + [1] * n_per_class)
    return x, y


def test_dense_pca_representation_range_and_shape():
    x, y = _fake_image_pool(n_per_class=40, img_size=8)
    raw = stratified_split(x, y, class_pos=1, class_neg=0, n_train=20, n_val=10, n_test=10, seed=0)
    rep = dense_pca_representation(raw, n_components=4, seed=0)

    assert rep["X_train"].shape == (40, 4)
    assert rep["X_train"].min() >= -1e-9
    assert rep["X_train"].max() <= np.pi + 1e-9


def test_amplitude_representation_is_unit_norm():
    x, y = _fake_image_pool(n_per_class=40, img_size=28)
    raw = stratified_split(x, y, class_pos=1, class_neg=0, n_train=20, n_val=10, n_test=10, seed=0)
    rep = amplitude_representation(raw, img_size=28, pad_size=32)

    assert rep["X_train"].shape == (40, 32 * 32)
    norms = np.linalg.norm(rep["X_train"], axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-8)


def test_amplitude_representation_padding_is_centered_zero():
    x, y = _fake_image_pool(n_per_class=10, img_size=28)
    raw = stratified_split(x, y, class_pos=1, class_neg=0, n_train=5, n_val=2, n_test=2, seed=0)
    rep = amplitude_representation(raw, img_size=28, pad_size=32)

    img = rep["X_train"][0].reshape(32, 32)
    pad = (32 - 28) // 2
    assert np.allclose(img[:pad, :], 0.0)
    assert np.allclose(img[:, :pad], 0.0)
