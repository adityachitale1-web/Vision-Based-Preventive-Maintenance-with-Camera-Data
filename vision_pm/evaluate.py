"""
Evaluation (plan items 4 and 5): accuracy, precision, recall, F1, confusion matrix - plus the
checks that make those numbers trustworthy: confidence intervals, threshold and cost
analysis, calibration, baselines and a significance test.

Convention throughout: positive class = 1 = defective.
    precision = TP / (TP + FP)   of the parts we flag, how many are really defective
    recall    = TP / (TP + FN)   of the really defective parts, how many we catch
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true, y_pred = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    return {"TP": int(((y_true == 1) & (y_pred == 1)).sum()), "FP": int(((y_true == 0) & (y_pred == 1)).sum()),
            "FN": int(((y_true == 1) & (y_pred == 0)).sum()), "TN": int(((y_true == 0) & (y_pred == 0)).sum())}


def _safe(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def metrics_at(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> dict:
    c = confusion(y_true, proba >= threshold)
    p, r = _safe(c["TP"], c["TP"] + c["FP"]), _safe(c["TP"], c["TP"] + c["FN"])
    return {**c, "accuracy": _safe(c["TP"] + c["TN"], sum(c.values())), "precision": p, "recall": r,
            "f1": _safe(2 * p * r, p + r), "specificity": _safe(c["TN"], c["TN"] + c["FP"]),
            "false_alarm_rate": _safe(c["FP"], c["FP"] + c["TN"]), "miss_rate": _safe(c["FN"], c["FN"] + c["TP"])}


def ranking_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict:
    return {"roc_auc": float(roc_auc_score(y_true, proba)), "pr_auc": float(average_precision_score(y_true, proba)),
            "brier": float(brier_score_loss(y_true, proba))}


def bootstrap_ci(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5, n_boot: int = 1000,
                 seed: int = 0) -> pd.DataFrame:
    """95% percentile intervals: how much the test-set metrics could move on another sample."""
    rng = np.random.default_rng(seed)
    y_true, pred = np.asarray(y_true).astype(int), (np.asarray(proba) >= threshold).astype(int)
    n = len(y_true)
    idx = rng.integers(0, n, size=(n_boot, n))
    yt, yp = y_true[idx], pred[idx]
    tp = ((yt == 1) & (yp == 1)).sum(1).astype(float)
    fp = ((yt == 0) & (yp == 1)).sum(1).astype(float)
    fn = ((yt == 1) & (yp == 0)).sum(1).astype(float)
    tn = n - tp - fp - fn
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        rec = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    samples = {"accuracy": (tp + tn) / n, "precision": prec, "recall": rec, "f1": f1}
    point = metrics_at(y_true, proba, threshold)
    return pd.DataFrame([{"metric": k, "value": point[k], "ci_low": float(np.percentile(v, 2.5)),
                          "ci_high": float(np.percentile(v, 97.5))} for k, v in samples.items()])


def threshold_sweep(y_true: np.ndarray, proba: np.ndarray, cost_fp: float = 1.0, cost_fn: float = 10.0,
                    n: int = 99) -> pd.DataFrame:
    """Metrics and expected cost per 1,000 inspected parts at every decision threshold."""
    rows = []
    for t in np.linspace(0.01, 0.99, n):
        m = metrics_at(y_true, proba, t)
        rows.append({"threshold": round(float(t), 2), "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
                     "accuracy": m["accuracy"], "FP": m["FP"], "FN": m["FN"],
                     "cost_per_1000": 1000 * (cost_fp * m["FP"] + cost_fn * m["FN"]) / len(y_true)})
    return pd.DataFrame(rows)


def curves(y_true: np.ndarray, proba: np.ndarray) -> dict:
    fpr, tpr, _ = roc_curve(y_true, proba)
    prec, rec, _ = precision_recall_curve(y_true, proba)
    return {"fpr": fpr, "tpr": tpr, "precision": prec, "recall": rec}


def calibration_table(y_true: np.ndarray, proba: np.ndarray, bins: int = 8) -> pd.DataFrame:
    """Does 'P(defective) = 0.8' mean 80% of such parts are defective?"""
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(proba, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = which == b
        if m.sum():
            rows.append({"bin": f"{edges[b]:.2f}-{edges[b + 1]:.2f}", "mean_predicted": float(proba[m].mean()),
                         "observed_defect_rate": float(np.mean(y_true[m])), "n": int(m.sum())})
    return pd.DataFrame(rows)


def per_defect_recall(meta: pd.DataFrame, proba: np.ndarray, threshold: float = 0.5) -> pd.DataFrame:
    d = meta.assign(flagged=proba >= threshold)
    d = d[d["label"] == 1]
    out = d.groupby("defect_type").agg(n=("flagged", "size"), caught=("flagged", "sum"),
                                       mean_severity=("severity", "mean")).reset_index()
    out["recall"] = out["caught"] / out["n"]
    return out.sort_values("recall")


def recall_by_severity(meta: pd.DataFrame, proba: np.ndarray, threshold: float = 0.5, bins: int = 5) -> pd.DataFrame:
    d = meta.assign(flagged=proba >= threshold)
    d = d[d["label"] == 1].copy()
    if d.empty:
        return pd.DataFrame(columns=["severity", "n", "recall"])
    d["bin"] = pd.cut(d["severity"], bins)
    out = d.groupby("bin", observed=True).agg(n=("flagged", "size"), recall=("flagged", "mean"),
                                              severity=("severity", "mean")).reset_index(drop=True)
    return out[["severity", "n", "recall"]]


# --------------------------------------------------------------------------- #
# Baselines - is a CNN actually needed?
# --------------------------------------------------------------------------- #
def _downsample(X: np.ndarray, factor: int = 4) -> np.ndarray:
    n, h, w, _ = X.shape
    return X[..., 0].reshape(n, h // factor, factor, w // factor, factor).mean(axis=(2, 4)).reshape(n, -1)


def baseline_probabilities(X_train, y_train, X_test, seed: int = 0) -> dict:
    """Two reference models: always-predict-majority and logistic regression on raw pixels."""
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=400, C=0.05, random_state=seed))
    lr.fit(_downsample(X_train), y_train)
    return {"Majority class (always 'normal')": np.zeros(len(X_test)),
            "Logistic regression on pixels": lr.predict_proba(_downsample(X_test))[:, 1]}


def mcnemar(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """Exact McNemar test on the parts where two models disagree about being right."""
    a_ok, b_ok = pred_a == y_true, pred_b == y_true
    only_a, only_b = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
    p = float(binomtest(only_a, only_a + only_b, 0.5).pvalue) if only_a + only_b else 1.0
    return {"only_a_correct": only_a, "only_b_correct": only_b, "p_value": p}


# --------------------------------------------------------------------------- #
# Statistical process control on the inspection results (p-chart)
# --------------------------------------------------------------------------- #
def p_chart(flagged_counts: np.ndarray, batch_size: int, baseline_batches: int) -> pd.DataFrame:
    """Proportion chart: centre line and 3-sigma limits from the first `baseline_batches`."""
    p = np.asarray(flagged_counts) / batch_size
    p_bar = float(p[:baseline_batches].mean())
    sigma = float(np.sqrt(p_bar * (1 - p_bar) / batch_size))
    ucl, lcl = p_bar + 3 * sigma, max(0.0, p_bar - 3 * sigma)
    return pd.DataFrame({"batch": np.arange(1, len(p) + 1), "flagged_rate": p, "centre": p_bar, "ucl": ucl,
                         "lcl": lcl, "out_of_control": (p > ucl) | (p < lcl),
                         "phase": np.where(np.arange(len(p)) < baseline_batches, "baseline", "monitoring")})
