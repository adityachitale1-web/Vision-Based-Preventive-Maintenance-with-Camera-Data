"""Stress tests and the production-line simulation. Both return raw probabilities so the
dashboard can re-apply any decision threshold without re-running the model."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .data import DataConfig, generate_images
from .model import predict_proba

NOISE_LEVELS = (0.0, 0.02, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.14, 0.20)
LIGHTING_LEVELS = (0.0, 0.2, 0.5, 0.8, 1.2)
SEVERITY_LEVELS = (0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0)


def stress_test(model, cfg: DataConfig, n: int = 400, seed: int = 7) -> dict:
    """Fresh images the model has never seen, under conditions that drift away from training."""
    out = {"noise": [], "lighting": [], "severity": []}
    for i, v in enumerate(NOISE_LEVELS):
        X, meta = generate_images(n, replace(cfg, noise_sigma=v, noise_max=None), np.random.default_rng(seed + i))
        out["noise"].append({"value": v, "y": meta["label"].to_numpy(), "proba": predict_proba(model, X)})
    for i, v in enumerate(LIGHTING_LEVELS):
        X, meta = generate_images(n, replace(cfg, lighting=v), np.random.default_rng(seed + 100 + i))
        out["lighting"].append({"value": v, "y": meta["label"].to_numpy(), "proba": predict_proba(model, X)})
    for i, v in enumerate(SEVERITY_LEVELS):                       # defective parts only -> recall
        c = replace(cfg, severity_min=v, severity_max=v, defect_rate=1.0)
        X, meta = generate_images(n // 2, c, np.random.default_rng(seed + 200 + i))
        out["severity"].append({"value": v, "y": meta["label"].to_numpy(), "proba": predict_proba(model, X)})
    return out


def simulate_line(model, cfg: DataConfig, n_batches: int = 30, batch_size: int = 100, base_rate: float = 0.04,
                  wear_start: int = 19, final_rate: float = 0.20, seed: int = 11) -> dict:
    """A production run: a healthy tool, then progressive tool wear that raises the defect rate."""
    rng = np.random.default_rng(seed)
    b = np.arange(1, n_batches + 1)
    ramp = np.clip((b - wear_start + 1) / max(n_batches - wear_start + 1, 1), 0, 1)
    process_rate = base_rate + (final_rate - base_rate) * ramp
    xs, ys = [], []
    for rate in process_rate:
        k = int(rng.binomial(batch_size, rate))
        X, meta = generate_images(batch_size, cfg, rng, defect_rate=k / batch_size)
        xs.append(X)
        ys.append(meta["label"].to_numpy())
    proba = predict_proba(model, np.concatenate(xs)).reshape(n_batches, batch_size)
    y = np.stack(ys)
    return {"process_rate": process_rate, "actual_rate": y.mean(axis=1), "y": y, "proba": proba,
            "wear_start": wear_start, "batch_size": batch_size}
