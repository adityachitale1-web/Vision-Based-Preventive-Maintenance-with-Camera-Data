import numpy as np
import pandas as pd
import pytest

from vision_pm import evaluate as ev

Y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
P = np.array([0.9, 0.8, 0.6, 0.2, 0.7, 0.4, 0.3, 0.2, 0.1, 0.05])


def test_metrics_match_hand_calculation():
    m = ev.metrics_at(Y, P, 0.5)
    assert (m["TP"], m["FP"], m["FN"], m["TN"]) == (3, 1, 1, 5)
    assert m["accuracy"] == pytest.approx(0.8)
    assert m["precision"] == pytest.approx(0.75)
    assert m["recall"] == pytest.approx(0.75)
    assert m["f1"] == pytest.approx(0.75)
    assert m["false_alarm_rate"] == pytest.approx(1 / 6)


def test_no_flags_gives_zero_precision_and_recall_without_error():
    m = ev.metrics_at(Y, np.zeros(len(Y)), 0.5)
    assert m["precision"] == 0 and m["recall"] == 0 and m["f1"] == 0
    assert m["accuracy"] == pytest.approx(0.6)


def test_lower_threshold_never_lowers_recall():
    sweep = ev.threshold_sweep(Y, P)
    assert (np.diff(sweep["recall"].to_numpy()) <= 1e-12).all()


def test_cost_uses_both_error_costs():
    sweep = ev.threshold_sweep(Y, P, cost_fp=1.0, cost_fn=10.0)
    row = sweep.loc[(sweep["threshold"] - 0.5).abs().idxmin()]
    assert row["cost_per_1000"] == pytest.approx((1 * 1 + 1 * 10) / len(Y) * 1000)
    cheap_misses = ev.threshold_sweep(Y, P, cost_fp=10.0, cost_fn=1.0)
    best_a = sweep.loc[sweep["cost_per_1000"].idxmin(), "threshold"]
    best_b = cheap_misses.loc[cheap_misses["cost_per_1000"].idxmin(), "threshold"]
    assert best_a < best_b            # expensive misses -> lower threshold


def test_bootstrap_interval_contains_the_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    p = np.clip(y * 0.6 + rng.normal(0.2, 0.2, 400), 0, 1)
    ci = ev.bootstrap_ci(y, p, 0.5, n_boot=200)
    assert set(ci["metric"]) == {"accuracy", "precision", "recall", "f1"}
    assert ((ci["ci_low"] <= ci["value"]) & (ci["value"] <= ci["ci_high"])).all()


def test_ranking_metrics_for_a_perfect_and_a_useless_score():
    assert ev.ranking_metrics(Y, Y.astype(float))["roc_auc"] == pytest.approx(1.0)
    assert ev.ranking_metrics(Y, P)["roc_auc"] == pytest.approx(19.5 / 24)       # one tie counts half


def test_mcnemar_counts_disagreements():
    y = np.array([1] * 10 + [0] * 10)
    a = y.copy()
    b = y.copy()
    b[:8] = 0                                   # model b misses 8 that model a gets
    r = ev.mcnemar(y, a, b)
    assert (r["only_a_correct"], r["only_b_correct"]) == (8, 0)
    assert r["p_value"] < 0.05
    assert ev.mcnemar(y, a, a)["p_value"] == pytest.approx(1.0)


def test_p_chart_limits_and_alarm():
    flagged = np.array([8, 9, 7, 8, 8, 9, 7, 8, 30])
    pc = ev.p_chart(flagged, batch_size=100, baseline_batches=8)
    centre = 0.08
    sigma = np.sqrt(centre * (1 - centre) / 100)
    assert pc["centre"].iloc[0] == pytest.approx(centre)
    assert pc["ucl"].iloc[0] == pytest.approx(centre + 3 * sigma)
    assert pc["lcl"].iloc[0] == pytest.approx(max(centre - 3 * sigma, 0))
    assert pc["out_of_control"].tolist() == [False] * 8 + [True]


def test_per_defect_recall_and_severity():
    meta = pd.DataFrame({"label": [1, 1, 1, 1, 0, 0], "defect_type": ["pit", "pit", "dent", "dent", "none", "none"],
                         "severity": [0.9, 0.8, 0.4, 0.3, np.nan, np.nan]})
    proba = np.array([0.9, 0.9, 0.9, 0.1, 0.1, 0.1])
    r = ev.per_defect_recall(meta, proba, 0.5).set_index("defect_type")["recall"]
    assert r["pit"] == pytest.approx(1.0) and r["dent"] == pytest.approx(0.5)
