"""Train the two CNNs the dashboard ships with and save them to artifacts/.

    python train_baseline.py            # both
    python train_baseline.py baseline   # default settings (the model the dashboard opens with)
    python train_baseline.py robust     # same CNN trained on images with varying noise (stress-test comparison)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from vision_pm import evaluate as ev
from vision_pm.data import ROBUST_CONFIG, DataConfig, make_dataset
from vision_pm.model import ModelConfig, predict_proba, train_cnn

ART = Path(__file__).resolve().parent / "artifacts"


def train_and_save(name: str, dcfg: DataConfig) -> None:
    mcfg = ModelConfig()
    data = make_dataset(dcfg)
    model, info = train_cnn(data["X_train"], data["y_train"], mcfg,
                            progress=lambda e, n, logs: print(f"[{name}] epoch {e:2d}/{n}  loss {logs.get('loss', 0):.3f}  "
                                                              f"val_loss {logs.get('val_loss', 0):.3f}  "
                                                              f"val_acc {logs.get('val_accuracy', 0):.3f}"))
    proba = predict_proba(model, data["X_test"])
    m = {**ev.metrics_at(data["y_test"], proba), **ev.ranking_metrics(data["y_test"], proba)}
    ART.mkdir(exist_ok=True)
    model.save(ART / f"{name}_model.keras")
    (ART / f"{name}_info.json").write_text(json.dumps(
        {"data_config": dcfg.to_dict(), "model_config": mcfg.to_dict(), "training": info, "test_metrics_at_0.5": m},
        indent=2))
    print(f"[{name}] saved to {ART}")
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()})


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "baseline"):
        train_and_save("baseline", DataConfig())
    if which in ("all", "robust"):
        train_and_save("robust", ROBUST_CONFIG)


if __name__ == "__main__":
    main()
