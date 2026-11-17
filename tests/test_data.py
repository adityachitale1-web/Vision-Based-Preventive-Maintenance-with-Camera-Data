from dataclasses import replace

import numpy as np

from vision_pm.data import DEFECT_TYPES, DataConfig, generate_images, make_dataset, render_part

SMALL = DataConfig(n_images=400)


def test_images_have_the_right_shape_and_range():
    d = make_dataset(SMALL)
    assert d["X_train"].shape == (300, 64, 64, 1) and d["X_test"].shape == (100, 64, 64, 1)
    assert d["X_train"].dtype == np.float32
    assert 0.0 <= d["X_train"].min() and d["X_train"].max() <= 1.0


def test_two_classes_at_the_requested_rate():
    d = make_dataset(SMALL)
    y = np.concatenate([d["y_train"], d["y_test"]])
    assert set(np.unique(y)) == {0, 1}
    assert abs(y.mean() - SMALL.defect_rate) < 0.01
    assert abs(d["y_test"].mean() - SMALL.defect_rate) < 0.05          # stratified split


def test_same_seed_same_images_different_seed_different_images():
    a, b, c = make_dataset(SMALL), make_dataset(SMALL), make_dataset(replace(SMALL, seed=7))
    assert np.array_equal(a["X_test"], b["X_test"])
    assert not np.array_equal(a["X_test"], c["X_test"])


def test_metadata_matches_labels():
    d = make_dataset(SMALL)
    meta = d["meta_test"]
    assert (meta["label"].to_numpy() == d["y_test"]).all()
    assert (meta.loc[meta.label == 0, "defect_type"] == "none").all()
    assert set(meta.loc[meta.label == 1, "defect_type"]) <= set(DEFECT_TYPES)
    sev = meta.loc[meta.label == 1, "severity"]
    assert sev.between(SMALL.severity_min, SMALL.severity_max).all()
    assert meta.loc[meta.label == 0, "severity"].isna().all()


def test_every_defect_type_changes_the_image():
    cfg = replace(DataConfig(), noise_sigma=0.0)
    for defect in DEFECT_TYPES:
        clean, _ = render_part(np.random.default_rng(3), cfg, None)
        flawed, meta = render_part(np.random.default_rng(3), cfg, defect, 1.0)
        assert meta["label"] == 1 and meta["defect_type"] == defect
        assert np.abs(clean - flawed).max() > 0.05, defect


def test_noise_setting_controls_pixel_noise():
    quiet, _ = generate_images(20, replace(DataConfig(), noise_sigma=0.0), np.random.default_rng(1))
    noisy, _ = generate_images(20, replace(DataConfig(), noise_sigma=0.15), np.random.default_rng(1))
    rough = lambda X: np.abs(np.diff(X[..., 0], axis=2)).mean()
    assert rough(noisy) > 3 * rough(quiet)


def test_only_selected_defect_types_are_generated():
    _, meta = generate_images(100, replace(DataConfig(), defect_types=("pit",)), np.random.default_rng(0))
    assert set(meta.loc[meta.label == 1, "defect_type"]) == {"pit"}
