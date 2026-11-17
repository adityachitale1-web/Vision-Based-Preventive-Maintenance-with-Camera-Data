"""
Synthetic camera images of machined parts (plan item 2: "Dataset").

Each image is a 64 x 64 greyscale view of a ring-shaped part (a washer / bearing race) on a
conveyor. Two classes:

    0 = normal      benign variation only
    1 = defective   one surface or shape defect

"Noise and shape distortions" are both present, as the exercise asks:

  noise              Gaussian sensor noise, uneven lighting, surface texture, slight blur
  shape distortions  position / radius / ovality vary in every part (benign); defective parts
                     get a notch, a dent (flat spot), a scratch, a pit or a crack

Normal parts also carry harmless distractors (faint tooling marks, dust specks) and defect
severity is drawn from a range, so some defects are subtle. That keeps the task from being
trivially separable - the lecture's warning about a perfect score on synthetic data.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from sklearn.model_selection import train_test_split

DEFECT_TYPES = ("notch", "dent", "scratch", "pit", "crack")
DEFECT_INFO = {
    "notch": "Triangular chip out of the outer edge (shape distortion)",
    "dent": "Flat spot on the rim (shape distortion)",
    "scratch": "Thin straight line across the surface",
    "pit": "One to three small dark craters",
    "crack": "Jagged dark line running in from the edge",
}


@dataclass(frozen=True)
class DataConfig:
    n_images: int = 3000
    image_size: int = 64
    defect_rate: float = 0.40          # share of defective parts in the dataset
    noise_sigma: float = 0.05          # Gaussian sensor noise (pixel scale 0-1)
    noise_max: float | None = None     # if set, each image draws its noise from U(noise_sigma, noise_max)
    severity_min: float = 0.30         # 0 = barely visible, 1 = obvious
    severity_max: float = 1.00
    lighting: float = 0.20             # strength of the uneven-illumination gradient
    defect_types: tuple = DEFECT_TYPES
    test_size: float = 0.25
    seed: int = 42

    def to_dict(self) -> dict:
        d = asdict(self)
        d["defect_types"] = list(self.defect_types)
        return d


# Training recipe for the noise-robust comparison model: every image draws its own noise level,
# and the set is twice as large because the task is harder.
ROBUST_CONFIG = DataConfig(n_images=6000, noise_sigma=0.03, noise_max=0.10)


# --------------------------------------------------------------------------- #
# One part
# --------------------------------------------------------------------------- #
def _grid(size: int):
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    return xx, yy


def _segment_distance(xx, yy, x0, y0, x1, y1):
    """Distance from every pixel to the segment (x0,y0)-(x1,y1)."""
    vx, vy = x1 - x0, y1 - y0
    t = np.clip(((xx - x0) * vx + (yy - y0) * vy) / (vx * vx + vy * vy + 1e-6), 0, 1)
    return np.hypot(xx - (x0 + t * vx), yy - (y0 + t * vy))


def render_part(rng: np.random.Generator, cfg: DataConfig, defect: str | None = None,
                severity: float | None = None) -> tuple[np.ndarray, dict]:
    """Return one image (H, W) in [0, 1] and its metadata."""
    S = cfg.image_size
    xx, yy = _grid(S)

    # ---- geometry: benign shape variation in EVERY part ------------------------------
    cx, cy = S / 2 + rng.uniform(-0.04, 0.04, 2) * S
    r_out = rng.uniform(0.36, 0.43) * S
    r_in = rng.uniform(0.28, 0.42) * r_out
    ovality = rng.uniform(-0.05, 0.05)
    theta = rng.uniform(0, np.pi)
    dx, dy = xx - cx, yy - cy
    xr = dx * np.cos(theta) + dy * np.sin(theta)
    yr = -dx * np.sin(theta) + dy * np.cos(theta)
    d = np.hypot(xr / (1 + ovality), yr / (1 - ovality))        # elliptical radius
    ang = np.arctan2(dy, dx)
    mask = np.clip(r_out - d + 0.5, 0, 1) * np.clip(d - r_in + 0.5, 0, 1)   # soft-edged ring

    # ---- surface: brightness, machining rings, low-frequency texture ------------------
    bg = rng.uniform(0.12, 0.28)
    fg = rng.uniform(0.58, 0.84)
    texture = gaussian_filter(rng.standard_normal((S, S)).astype(np.float32), 3.0)
    texture /= (np.abs(texture).max() + 1e-6)
    rings = np.sin(d * rng.uniform(0.9, 1.6) + rng.uniform(0, 2 * np.pi))
    surface = fg * (1 + 0.04 * texture + 0.02 * rings)

    # ---- harmless distractors, in both classes ------------------------------------------
    if rng.random() < 0.35:                                     # faint tooling mark
        a = rng.uniform(0, 2 * np.pi)
        rr = rng.uniform(r_in + 1, r_out - 1)
        x0, y0 = cx + rr * np.cos(a), cy + rr * np.sin(a)
        b = a + rng.uniform(0.3, 0.8)
        x1, y1 = cx + rr * np.cos(b), cy + rr * np.sin(b)
        line = np.clip(1.0 - _segment_distance(xx, yy, x0, y0, x1, y1), 0, 1)
        surface -= fg * rng.uniform(0.02, 0.05) * line
    if rng.random() < 0.35:                                     # dust speck
        a, rr = rng.uniform(0, 2 * np.pi), rng.uniform(r_in, r_out)
        speck = np.exp(-((xx - cx - rr * np.cos(a)) ** 2 + (yy - cy - rr * np.sin(a)) ** 2) / 1.2)
        surface += fg * rng.uniform(0.04, 0.09) * speck

    # ---- defect -------------------------------------------------------------------------
    s = float(severity) if severity is not None else float(rng.uniform(cfg.severity_min, cfg.severity_max))
    phi = rng.uniform(0, 2 * np.pi)
    if defect == "notch":                                       # wedge that opens outward
        depth = (0.14 + 0.34 * s) * r_out
        half_w = 0.10 + 0.20 * s
        da = np.abs(np.angle(np.exp(1j * (ang - phi))))
        inside = (d > r_out - depth) & (da < half_w * (d - (r_out - depth)) / depth)
        mask = np.where(inside, 0.0, mask)
    elif defect == "dent":                                      # chord cut = flat spot
        depth = (0.06 + 0.17 * s) * r_out
        proj = dx * np.cos(phi) + dy * np.sin(phi)
        mask = mask * np.clip((r_out - depth) - proj + 0.5, 0, 1)
    elif defect == "scratch":
        length = (0.50 + 0.90 * s) * r_out
        rr = rng.uniform(r_in, r_out) * 0.9
        mx, my = cx + rr * np.cos(phi), cy + rr * np.sin(phi)
        a = rng.uniform(0, np.pi)
        x0, y0 = mx - length / 2 * np.cos(a), my - length / 2 * np.sin(a)
        x1, y1 = mx + length / 2 * np.cos(a), my + length / 2 * np.sin(a)
        width = 0.9 + 0.7 * s
        line = np.clip(width - _segment_distance(xx, yy, x0, y0, x1, y1), 0, 1)
        sign = -1.0 if rng.random() < 0.7 else 1.0              # mostly dark, sometimes bright
        surface += sign * fg * (0.14 + 0.34 * s) * line
    elif defect == "pit":
        for _ in range(int(rng.integers(1, 4))):
            a, rr = rng.uniform(0, 2 * np.pi), rng.uniform(r_in + 1.5, r_out - 1.5)
            px, py = cx + rr * np.cos(a), cy + rr * np.sin(a)
            rad = 1.3 + 2.4 * s
            blob = np.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2 * rad ** 2))
            surface -= fg * (0.20 + 0.40 * s) * blob
    elif defect == "crack":
        x, y = cx + r_out * np.cos(phi), cy + r_out * np.sin(phi)
        heading = phi + np.pi + rng.uniform(-0.3, 0.3)          # runs inward from the rim
        step = (0.55 + 0.80 * s) * r_out / 5
        crack = np.zeros_like(surface)
        for _ in range(5):
            heading += rng.uniform(-0.55, 0.55)
            nx, ny = x + step * np.cos(heading), y + step * np.sin(heading)
            crack = np.maximum(crack, np.clip(1.2 - _segment_distance(xx, yy, x, y, nx, ny), 0, 1))
            x, y = nx, ny
        surface -= fg * (0.18 + 0.36 * s) * crack
    elif defect is not None:
        raise ValueError(f"unknown defect type '{defect}'")

    # ---- camera: compose, uneven light, focus blur, sensor noise --------------------------
    img = bg * (1 - mask) + surface * mask
    g = rng.uniform(0, 2 * np.pi)
    light = 1 + cfg.lighting * rng.uniform(0.3, 1.0) * ((xx - S / 2) * np.cos(g) + (yy - S / 2) * np.sin(g)) / S
    img = gaussian_filter(img * light, 0.6)
    sigma = cfg.noise_sigma if cfg.noise_max is None else rng.uniform(cfg.noise_sigma, cfg.noise_max)
    img = img + rng.normal(0, sigma, img.shape)
    img = np.clip(img, 0, 1).astype(np.float32)
    meta = {"label": int(defect is not None), "defect_type": defect or "none",
            "severity": s if defect else np.nan, "radius_px": float(r_out), "ovality": float(ovality),
            "surface_brightness": float(fg)}
    return img, meta


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
def generate_images(n: int, cfg: DataConfig, rng: np.random.Generator,
                    defect_rate: float | None = None) -> tuple[np.ndarray, pd.DataFrame]:
    """n images with the configured class mix. Returns X (n, H, W, 1) and a metadata frame."""
    rate = cfg.defect_rate if defect_rate is None else defect_rate
    if not cfg.defect_types:
        raise ValueError("at least one defect type is required")
    n_def = int(round(n * rate))
    labels = np.array([1] * n_def + [0] * (n - n_def))
    rng.shuffle(labels)
    types = rng.choice(cfg.defect_types, size=n)
    X = np.empty((n, cfg.image_size, cfg.image_size, 1), dtype=np.float32)
    rows = []
    for i in range(n):
        img, meta = render_part(rng, cfg, defect=str(types[i]) if labels[i] else None)
        X[i, ..., 0] = img
        rows.append(meta)
    return X, pd.DataFrame(rows)


def make_dataset(cfg: DataConfig) -> dict:
    """Generate the full dataset and split it. The test set is held out before any training."""
    rng = np.random.default_rng(cfg.seed)
    X, meta = generate_images(cfg.n_images, cfg, rng)
    y = meta["label"].to_numpy()
    idx = np.arange(len(y))
    # stratify on defect type so every defect appears in train and test in proportion
    tr, te = train_test_split(idx, test_size=cfg.test_size, random_state=cfg.seed, stratify=meta["defect_type"])
    return {"X_train": X[tr], "y_train": y[tr], "meta_train": meta.iloc[tr].reset_index(drop=True),
            "X_test": X[te], "y_test": y[te], "meta_test": meta.iloc[te].reset_index(drop=True),
            "config": cfg}


def shifted_test_set(cfg: DataConfig, n: int, seed: int, **overrides) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """A fresh test set drawn under different conditions (for the stress test)."""
    from dataclasses import replace
    c = replace(cfg, **overrides)
    X, meta = generate_images(n, c, np.random.default_rng(seed))
    return X, meta["label"].to_numpy(), meta
