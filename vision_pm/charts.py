"""Plotly figures for the dashboard. One y-axis per chart; colour follows the series, never its rank."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Categorical slots (validated colour-blind-safe order), status colours and a one-hue blue ramp.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GOOD, CRITICAL, MUTED = "#0ca30c", "#d03b3b", "#8a8985"
BLUE_RAMP = [[0.0, "#cde2fb"], [0.25, "#86b6ef"], [0.5, "#3987e5"], [0.75, "#1c5cab"], [1.0, "#0d366b"]]


def _layout(fig: go.Figure, title: str, x: str, y: str, height: int = 340, legend: bool = True) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=15), x=0, xanchor="left", y=0.98, yanchor="top", pad=dict(l=6, t=4)),
        height=height, margin=dict(l=10, r=10, t=74 if legend else 44, b=10),
        xaxis_title=x, yaxis_title=y, showlegend=legend, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0))
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.18)", zeroline=False)
    return fig


def training_curve(history: dict, key: str, title: str, y: str, best_epoch: int,
                   note_position: str = "bottom right") -> go.Figure:
    ep = list(range(1, len(history[key]) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ep, y=history[key], name="Training", mode="lines+markers",
                             line=dict(color=BLUE, width=2), marker=dict(size=6)))
    fig.add_trace(go.Scatter(x=ep, y=history[f"val_{key}"], name="Validation", mode="lines+markers",
                             line=dict(color=ORANGE, width=2), marker=dict(size=6)))
    fig.add_vline(x=best_epoch, line=dict(color=MUTED, width=1, dash="dot"),
                  annotation_text=f"best epoch {best_epoch}", annotation_position=note_position)
    return _layout(fig, title, "Epoch", y)


def confusion_matrix(c: dict) -> go.Figure:
    z = np.array([[c["TN"], c["FP"]], [c["FN"], c["TP"]]])
    row_pct = z / np.maximum(z.sum(axis=1, keepdims=True), 1)
    names = [["True negative", "False alarm (FP)"], ["Missed defect (FN)", "True positive"]]
    text = [[f"<b>{z[i, j]}</b><br>{names[i][j]}<br>{row_pct[i, j]:.1%} of row" for j in range(2)] for i in range(2)]
    fig = go.Figure(go.Heatmap(z=z, x=["Predicted<br>normal", "Predicted<br>defective"], y=["Actual<br>normal", "Actual<br>defective"],
                               colorscale=BLUE_RAMP, showscale=False, xgap=3, ygap=3, hoverinfo="skip"))
    hi = z.max()
    for i in range(2):
        for j in range(2):
            fig.add_annotation(x=j, y=i, text=text[i][j], showarrow=False,
                               font=dict(size=13, color="white" if z[i, j] > 0.45 * hi else "#0b0b0b"))
    fig = _layout(fig, "Confusion matrix (test set)", "", "", height=360, legend=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_layout(hovermode=False)
    return fig


def roc(curve: dict, auc: float, point: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve["fpr"], y=curve["tpr"], name=f"CNN (AUC {auc:.3f})", mode="lines",
                             line=dict(color=BLUE, width=2)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random guess (AUC 0.5)", mode="lines",
                             line=dict(color=MUTED, width=1, dash="dash")))
    fig.add_trace(go.Scatter(x=[point["false_alarm_rate"]], y=[point["recall"]], name="Current threshold",
                             mode="markers", marker=dict(color=ORANGE, size=11, line=dict(color="white", width=2))))
    fig = _layout(fig, "ROC curve", "False positive rate (normal parts flagged)", "True positive rate (recall)")
    fig.update_layout(hovermode="closest")
    return fig


def precision_recall(curve: dict, ap: float, point: dict, prevalence: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve["recall"], y=curve["precision"], name=f"CNN (AP {ap:.3f})",
                             mode="lines", line=dict(color=BLUE, width=2)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[prevalence, prevalence], name=f"Flag everything ({prevalence:.0%})",
                             mode="lines", line=dict(color=MUTED, width=1, dash="dash")))
    fig.add_trace(go.Scatter(x=[point["recall"]], y=[point["precision"]], name="Current threshold", mode="markers",
                             marker=dict(color=ORANGE, size=11, line=dict(color="white", width=2))))
    fig = _layout(fig, "Precision-recall curve (AP = average precision)", "Recall (defects caught)", "Precision (flags that are real)")
    fig.update_layout(hovermode="closest")
    fig.update_yaxes(range=[0, 1.03])
    return fig


def threshold_metrics(sweep: pd.DataFrame, threshold: float) -> go.Figure:
    fig = go.Figure()
    for col, name, colour in (("precision", "Precision", BLUE), ("recall", "Recall", ORANGE), ("f1", "F1 score", AQUA)):
        fig.add_trace(go.Scatter(x=sweep["threshold"], y=sweep[col], name=name, mode="lines",
                                 line=dict(color=colour, width=2)))
    fig.add_vline(x=threshold, line=dict(color=MUTED, width=1, dash="dot"),
                  annotation_text=f"current {threshold:.2f}", annotation_position="bottom right")
    fig = _layout(fig, "Precision, recall and F1 at every threshold", "Decision threshold on P(defective)", "Score")
    fig.update_yaxes(range=[0, 1.03])
    return fig


def cost_curve(sweep: pd.DataFrame, threshold: float, best_threshold: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sweep["threshold"], y=sweep["cost_per_1000"], name="Expected cost", mode="lines",
                             line=dict(color=BLUE, width=2)))
    best = sweep.loc[(sweep["threshold"] - best_threshold).abs().idxmin()]
    fig.add_trace(go.Scatter(x=[best["threshold"]], y=[best["cost_per_1000"]], mode="markers",
                             marker=dict(color=GOOD, size=11, line=dict(color="white", width=2)), name="Lowest cost"))
    fig.add_annotation(x=best["threshold"], y=best["cost_per_1000"], text=f"lowest cost at {best['threshold']:.2f}",
                       showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor=MUTED, ax=70, ay=-70)
    fig.add_vline(x=threshold, line=dict(color=MUTED, width=1, dash="dot"),
                  annotation_text=f"current {threshold:.2f}", annotation_position="top right")
    return _layout(fig, "Expected cost of errors per 1,000 parts inspected", "Decision threshold on P(defective)",
                   "Cost units", legend=False)


def score_histogram(proba: np.ndarray, y: np.ndarray, threshold: float) -> go.Figure:
    bins = dict(start=0, end=1, size=0.05)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=proba[y == 0], name="Actual normal", xbins=bins, marker_color=BLUE, opacity=0.8))
    fig.add_trace(go.Histogram(x=proba[y == 1], name="Actual defective", xbins=bins, marker_color=ORANGE, opacity=0.8))
    fig.add_vline(x=threshold, line=dict(color=MUTED, width=1, dash="dot"),
                  annotation_text=f"threshold {threshold:.2f}", annotation_position="top right")
    fig = _layout(fig, "Model output by true class", "Predicted P(defective)", "Parts in test set")
    fig.update_layout(barmode="group", bargap=0.15)
    return fig


def calibration(table: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Perfectly calibrated", mode="lines",
                             line=dict(color=MUTED, width=1, dash="dash")))
    fig.add_trace(go.Scatter(x=table["mean_predicted"], y=table["observed_defect_rate"], name="CNN",
                             mode="lines+markers", line=dict(color=BLUE, width=2), marker=dict(size=8),
                             customdata=table["n"], hovertemplate="predicted %{x:.2f}<br>observed %{y:.2f}<br>n = %{customdata}"))
    fig = _layout(fig, "Calibration: predicted probability vs. what actually happened", "Mean predicted P(defective)",
                  "Observed share defective")
    fig.update_layout(hovermode="closest")
    return fig


def horizontal_bars(df: pd.DataFrame, label: str, value: str, title: str, x_title: str, n_col: str | None = None) -> go.Figure:
    text = [f"{v:.0%}" + (f"  (n = {n})" if n_col else "") for v, n in zip(df[value], df[n_col] if n_col else df[value])]
    fig = go.Figure(go.Bar(x=df[value], y=df[label], orientation="h", marker_color=BLUE, text=text,
                           textposition="outside", cliponaxis=False, hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
    fig = _layout(fig, title, x_title, "", height=90 + 38 * len(df), legend=False)
    fig.update_xaxes(range=[0, 1.18], tickformat=".0%")
    fig.update_layout(hovermode="closest", bargap=0.35)
    fig.update_yaxes(showgrid=False)
    return fig


def stress_lines(df: pd.DataFrame, x: str, title: str, x_title: str, trained_at: float | None = None,
                 cols=(("recall", "Recall", ORANGE), ("precision", "Precision", BLUE), ("accuracy", "Accuracy", AQUA))) -> go.Figure:
    fig = go.Figure()
    for col, name, colour in cols:
        if col in df:
            fig.add_trace(go.Scatter(x=df[x], y=df[col], name=name, mode="lines+markers",
                                     line=dict(color=colour, width=2), marker=dict(size=7)))
    if trained_at is not None:
        fig.add_vline(x=trained_at, line=dict(color=MUTED, width=1, dash="dot"),
                      annotation_text="training conditions", annotation_position="bottom right")
    fig = _layout(fig, title, x_title, "Score")
    fig.update_yaxes(range=[0, 1.03])
    return fig


def spc_chart(df: pd.DataFrame, true_rate: np.ndarray) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["batch"], y=true_rate, name="True defect rate (unknown on a real line)",
                             mode="lines", line=dict(color=MUTED, width=1)))
    fig.add_trace(go.Scatter(x=df["batch"], y=df["flagged_rate"], name="Share flagged by the CNN",
                             mode="lines+markers", line=dict(color=BLUE, width=2), marker=dict(size=7)))
    ooc = df[df["out_of_control"]]
    if len(ooc):
        fig.add_trace(go.Scatter(x=ooc["batch"], y=ooc["flagged_rate"], name="Out of control", mode="markers",
                                 marker=dict(color=CRITICAL, size=12, symbol="x", line=dict(color="white", width=1))))
    for col, name, dash in (("ucl", "Upper control limit", "dash"), ("centre", "Centre line", "dot"),
                            ("lcl", "Lower control limit", "dash")):
        fig.add_hline(y=float(df[col].iloc[0]), line=dict(color=MUTED if col == "centre" else CRITICAL, width=1, dash=dash),
                      annotation_text=f"{name} {df[col].iloc[0]:.1%}", annotation_position="right")
    fig = _layout(fig, "p-chart: share of parts flagged per batch", "Production batch", "Share of batch", height=400)
    top = max(float(df["flagged_rate"].max()), float(np.max(true_rate)), float(df["ucl"].iloc[0]))
    fig.update_yaxes(tickformat=".0%", range=[0, top * 1.15])
    fig.update_layout(margin=dict(r=160))
    return fig
