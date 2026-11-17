"""
CNN for binary defect classification (plan item 3: "Model").

Architecture, as the exercise asks: convolution + pooling blocks, then dense layers, then a
single sigmoid unit that outputs P(defective). Pixel values are standardised first with the
training-set mean and standard deviation (the "normalisation" preprocessing step from the
lecture); without it the network sits on a long plateau before it starts to learn. Dropout
and early stopping are the two regularisers the lecture recommends against over-fitting.
"""
from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@dataclass(frozen=True)
class ModelConfig:
    filters: tuple = (16, 32, 64)      # one Conv2D + MaxPooling2D block per entry
    dense_units: int = 64
    dropout: float = 0.30
    learning_rate: float = 1e-3
    epochs: int = 30
    batch_size: int = 32
    validation_split: float = 0.20     # same split the lab notebook uses
    early_stopping_patience: int = 6
    seed: int = 42

    def to_dict(self) -> dict:
        d = asdict(self)
        d["filters"] = list(self.filters)
        return d


def build_cnn(cfg: ModelConfig, input_shape: tuple, pixel_mean: float = 0.4, pixel_std: float = 0.25) -> keras.Model:
    inputs = keras.Input(shape=input_shape, name="camera_image")
    # standardise with TRAINING-set statistics only, stored inside the model
    x = layers.Normalization(mean=float(pixel_mean), variance=float(pixel_std) ** 2, name="standardise")(inputs)
    for i, f in enumerate(cfg.filters, start=1):
        x = layers.Conv2D(f, 3, padding="same", activation="relu", name=f"conv{i}")(x)
        x = layers.MaxPooling2D(2, name=f"pool{i}")(x)
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(cfg.dense_units, activation="relu", name="dense")(x)
    x = layers.Dropout(cfg.dropout, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="defect_probability")(x)
    model = keras.Model(inputs, outputs, name="defect_cnn")
    model.compile(optimizer=keras.optimizers.Adam(cfg.learning_rate), loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


class _Progress(keras.callbacks.Callback):
    def __init__(self, fn, epochs):
        super().__init__()
        self.fn, self.epochs = fn, epochs

    def on_epoch_end(self, epoch, logs=None):
        self.fn(epoch + 1, self.epochs, logs or {})


def train_cnn(X_train: np.ndarray, y_train: np.ndarray, cfg: ModelConfig, progress=None) -> tuple[keras.Model, dict]:
    """Train with a held-out validation slice; early stopping restores the best weights."""
    keras.utils.set_random_seed(cfg.seed)
    # shuffle once, then hold out the last slice for validation (stratified by construction of the shuffle)
    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(len(y_train))
    X, y = X_train[order], y_train[order]
    n_val = int(round(len(y) * cfg.validation_split))
    X_tr, y_tr, X_val, y_val = X[:-n_val], y[:-n_val], X[-n_val:], y[-n_val:]

    model = build_cnn(cfg, X.shape[1:], pixel_mean=X_tr.mean(), pixel_std=X_tr.std())
    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=cfg.early_stopping_patience,
                                               restore_best_weights=True)]
    if progress is not None:
        callbacks.append(_Progress(progress, cfg.epochs))
    t0 = time.time()
    hist = model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=cfg.epochs,
                     batch_size=cfg.batch_size, verbose=0, callbacks=callbacks)
    h = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    best = int(np.argmin(h["val_loss"])) + 1
    info = {"history": h, "epochs_run": len(h["loss"]), "best_epoch": best,
            "stopped_early": len(h["loss"]) < cfg.epochs, "train_seconds": round(time.time() - t0, 1),
            "n_train": int(len(y_tr)), "n_val": int(len(y_val)), "params": int(model.count_params())}
    return model, info


def predict_proba(model: keras.Model, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
    return model.predict(X, batch_size=batch_size, verbose=0).ravel()


def layer_table(model: keras.Model) -> pd.DataFrame:
    rows = []
    for layer in model.layers:
        try:
            shape = tuple(layer.output.shape)
        except Exception:
            shape = ()
        rows.append({"layer": layer.name, "type": layer.__class__.__name__,
                     "output shape": " x ".join(str(s) for s in shape[1:]) or "-",
                     "parameters": int(layer.count_params())})
    return pd.DataFrame(rows)


def grad_cam(model: keras.Model, X: np.ndarray, conv_layer: str | None = None) -> np.ndarray:
    """Grad-CAM heat maps (n, H, W) in [0, 1]: where the last conv layer supports 'defective'.

    The gradient is taken on the logit (the value before the sigmoid). On the probability itself the
    gradient vanishes for confident predictions and the map comes out empty.
    """
    if conv_layer is None:
        conv_layer = [l.name for l in model.layers if isinstance(l, layers.Conv2D)][-1]
    out_layer = model.layers[-1]
    grad_model = keras.Model(model.inputs, [model.get_layer(conv_layer).output, out_layer.input])
    kernel, bias = out_layer.kernel, out_layer.bias
    x = tf.convert_to_tensor(X, dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_out, features = grad_model(x, training=False)
        score = (tf.matmul(features, kernel) + bias)[:, 0]
    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(weights * conv_out, axis=-1))
    cam = tf.image.resize(cam[..., None], X.shape[1:3], method="bilinear")[..., 0].numpy()
    peak = cam.reshape(len(cam), -1).max(axis=1)[:, None, None]
    return cam / np.maximum(peak, 1e-8)
