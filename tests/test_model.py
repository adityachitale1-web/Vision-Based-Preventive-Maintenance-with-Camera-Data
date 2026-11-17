import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from vision_pm import evaluate as ev
from vision_pm.data import DataConfig, make_dataset
from vision_pm.model import ModelConfig, build_cnn, grad_cam, layer_table, predict_proba, train_cnn

ART = Path(__file__).resolve().parents[1] / "artifacts"


def test_architecture_has_convolution_pooling_dense_and_a_sigmoid_output():
    model = build_cnn(ModelConfig(), (64, 64, 1))
    kinds = [type(l).__name__ for l in model.layers]
    assert kinds.count("Conv2D") == 3 and kinds.count("MaxPooling2D") == 3 and kinds.count("Dense") == 2
    assert "Dropout" in kinds
    assert model.output_shape == (None, 1)
    assert model.layers[-1].activation.__name__ == "sigmoid"
    assert model.loss == "binary_crossentropy"
    assert len(layer_table(model)) == len(model.layers)


def test_short_training_run_returns_history_and_probabilities():
    d = make_dataset(DataConfig(n_images=240))
    seen = []
    model, info = train_cnn(d["X_train"], d["y_train"], replace(ModelConfig(), epochs=2, filters=(4, 8, 8), dense_units=8),
                            progress=lambda e, n, logs: seen.append(e))
    assert seen == [1, 2] and info["epochs_run"] == 2
    assert set(info["history"]) >= {"loss", "val_loss", "accuracy", "val_accuracy"}
    assert info["n_train"] + info["n_val"] == len(d["X_train"])
    p = predict_proba(model, d["X_test"])
    assert p.shape == (len(d["X_test"]),) and (p >= 0).all() and (p <= 1).all()
    cam = grad_cam(model, d["X_test"][:3])
    assert cam.shape == (3, 64, 64) and cam.min() >= 0 and cam.max() <= 1


@pytest.mark.skipif(not (ART / "baseline_model.keras").exists(), reason="baseline artifact not present")
def test_saved_baseline_reproduces_its_recorded_test_metrics():
    from tensorflow import keras
    recorded = json.loads((ART / "baseline_info.json").read_text())["test_metrics_at_0.5"]
    d = make_dataset(DataConfig())
    proba = predict_proba(keras.models.load_model(ART / "baseline_model.keras"), d["X_test"])
    m = ev.metrics_at(d["y_test"], proba, 0.5)
    for k in ("accuracy", "precision", "recall", "f1"):
        assert m[k] == pytest.approx(recorded[k], abs=0.005), k
    assert m["accuracy"] > 0.85                       # clearly better than the 60% majority class
    assert m["accuracy"] < 1.0                        # and not the trivially separable case


@pytest.mark.skipif(not (ART / "baseline_model.keras").exists(), reason="baseline artifact not present")
def test_grad_cam_is_not_empty_for_confident_detections():
    from tensorflow import keras
    model = keras.models.load_model(ART / "baseline_model.keras")
    d = make_dataset(DataConfig())
    proba = predict_proba(model, d["X_test"])
    sure = np.where(proba > 0.999)[0][:5]
    assert len(sure) > 0
    assert (grad_cam(model, d["X_test"][sure]).reshape(len(sure), -1).max(axis=1) > 0.99).all()
