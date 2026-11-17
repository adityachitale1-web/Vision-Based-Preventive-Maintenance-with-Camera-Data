"""Headless run of the whole dashboard (slow: loads TensorFlow)."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def _metric(at, label):
    return next(m.value for m in at.metric if m.label == label)


def test_dashboard_renders_every_tab_with_the_baseline():
    at = AppTest.from_file(APP, default_timeout=600).run()
    assert not at.exception
    assert len(at.tabs) == 9
    assert _metric(at, "Accuracy") == "92.4%"
    assert _metric(at, "Recall") == "87.3%"


def test_lower_threshold_raises_recall():
    at = AppTest.from_file(APP, default_timeout=600).run()
    before = float(_metric(at, "Recall").rstrip("%"))
    at.sidebar.slider[0].set_value(0.15).run()
    assert not at.exception
    assert float(_metric(at, "Recall").rstrip("%")) > before


def test_training_a_new_model_from_the_sidebar():
    at = AppTest.from_file(APP, default_timeout=900).run()
    labels = {s.label: s for s in at.sidebar.slider}
    labels["Images"].set_value(1000)
    labels["Maximum epochs"].set_value(5)
    next(b for b in at.sidebar.button if b.label == "Generate data and train CNN").click()
    at.run()
    assert not at.exception
    assert any("Trained in this session" in i.value for i in at.info)
    assert _metric(at, "Images") == "1,000"
