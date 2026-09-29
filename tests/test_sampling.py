import numpy as np

from qcnn_benchmark.data.sampling import stratified_split


def _fake_pool(n_per_class=50, n_classes=3, n_features=4, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_per_class * n_classes, n_features))
    y = np.repeat(np.arange(n_classes), n_per_class)
    return x, y


def test_stratified_split_shapes_and_balance():
    x, y = _fake_pool(n_per_class=50)
    rep = stratified_split(x, y, class_pos=1, class_neg=0, n_train=10, n_val=5, n_test=5, seed=0)

    assert rep["X_train"].shape == (20, 4)
    assert rep["X_val"].shape == (10, 4)
    assert rep["X_test"].shape == (10, 4)
    for split in ("train", "val", "test"):
        labels = rep[f"y_{split}"]
        assert (labels == 1).sum() == (labels == 0).sum()


def test_stratified_split_label_convention():
    x, y = _fake_pool(n_per_class=50)
    rep = stratified_split(x, y, class_pos=2, class_neg=0, n_train=10, n_val=5, n_test=5, seed=0)
    assert set(np.unique(rep["y_train"])) == {0.0, 1.0}


def test_stratified_split_deterministic():
    x, y = _fake_pool(n_per_class=50)
    rep_a = stratified_split(x, y, class_pos=1, class_neg=0, n_train=10, n_val=5, n_test=5, seed=42)
    rep_b = stratified_split(x, y, class_pos=1, class_neg=0, n_train=10, n_val=5, n_test=5, seed=42)
    np.testing.assert_array_equal(rep_a["X_train"], rep_b["X_train"])


def test_stratified_split_raises_when_pool_too_small():
    x, y = _fake_pool(n_per_class=5)
    try:
        stratified_split(x, y, class_pos=1, class_neg=0, n_train=10, n_val=5, n_test=5, seed=0)
    except AssertionError:
        return
    raise AssertionError("se esperaba AssertionError por pool insuficiente")
