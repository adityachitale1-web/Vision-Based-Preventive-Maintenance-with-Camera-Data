"""
Vision-Based Preventive Maintenance with Camera Data (synthetic) - interactive dashboard.

Session IV lab exercise, "Reasoning and Decision-Making under Uncertainty" (Dr. Sandip Kumar Roy).
Run locally:  streamlit run app.py
"""
from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import streamlit as st

from vision_pm import charts
from vision_pm import evaluate as ev
from vision_pm.analysis import simulate_line, stress_test
from vision_pm.data import DEFECT_INFO, DEFECT_TYPES, DataConfig, make_dataset, render_part

ART = Path(__file__).resolve().parent / "artifacts"
DEFAULT_D, DEFAULT_M = DataConfig(), None      # ModelConfig is imported lazily with TensorFlow

st.set_page_config(page_title="Vision-Based Preventive Maintenance", page_icon="🔍", layout="wide")


# --------------------------------------------------------------------------- #
# Cached heavy lifting
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading TensorFlow…")
def tf_modules():
    from vision_pm import model as m
    return m


@st.cache_resource(show_spinner="Generating synthetic camera images…", max_entries=2)
def get_data(dcfg: DataConfig) -> dict:
    return make_dataset(dcfg)


@st.cache_resource(show_spinner="Loading the pre-trained baseline CNN…")
def load_baseline():
    from tensorflow import keras
    info = json.loads((ART / "baseline_info.json").read_text())
    return keras.models.load_model(ART / "baseline_model.keras"), info["training"]


@st.cache_resource(show_spinner="Fitting baseline models…", max_entries=2)
def get_baselines(dcfg: DataConfig) -> dict:
    d = get_data(dcfg)
    return ev.baseline_probabilities(d["X_train"], d["y_train"], d["X_test"])


@st.cache_resource(show_spinner="Running the stress test on fresh images…", max_entries=2)
def get_stress(dcfg: DataConfig, mcfg, _model) -> dict:
    return stress_test(_model, dcfg)


@st.cache_resource(show_spinner="Scoring the noise-robust CNN on the same images…", max_entries=2)
def get_robust_noise(dcfg: DataConfig):
    """Noise sweep for the comparison model trained on varying noise; None if it cannot be loaded."""
    try:
        from tensorflow import keras
        info = json.loads((ART / "robust_info.json").read_text())
        robust = keras.models.load_model(ART / "robust_model.keras")
    except Exception:
        return None
    return {"noise": stress_test(robust, dcfg)["noise"], "info": info}


@st.cache_resource(show_spinner="Simulating the production line…", max_entries=4)
def get_line(dcfg: DataConfig, mcfg, _model, n_batches, batch_size, base_rate, wear_start, final_rate) -> dict:
    return simulate_line(_model, dcfg, n_batches, batch_size, base_rate, wear_start, final_rate)


def is_baseline(dcfg, mcfg) -> bool:
    return dcfg == DataConfig() and mcfg == tf_modules().ModelConfig() and (ART / "baseline_model.keras").exists()


def get_run(dcfg: DataConfig, mcfg) -> dict:
    """Model + predictions for one (data, model) configuration, kept for the session."""
    runs = st.session_state.setdefault("runs", {})
    key = (dcfg, mcfg)
    if key in runs:
        return runs[key]
    M = tf_modules()
    data = get_data(dcfg)
    model = None
    if is_baseline(dcfg, mcfg):
        try:
            model, info = load_baseline()
            source = "Pre-trained baseline (shipped with the repo, reproducible with train_baseline.py)"
        except Exception as exc:                # e.g. a Keras version that cannot read the saved file
            st.warning(f"Could not load the saved baseline ({type(exc).__name__}); training it again instead.")
    if model is None:
        bar = st.progress(0.0, text="Training the CNN…")

        def tick(epoch, total, logs):
            bar.progress(epoch / total, text=f"Training the CNN… epoch {epoch}/{total} · "
                                             f"val accuracy {logs.get('val_accuracy', 0):.1%}")
        model, info = M.train_cnn(data["X_train"], data["y_train"], mcfg, progress=tick)
        bar.empty()
        source = "Trained in this session"
    run = {"model": model, "info": info, "source": source,
           "proba_test": M.predict_proba(model, data["X_test"]),
           "proba_train": M.predict_proba(model, data["X_train"])}
    if len(runs) >= 2:                          # bound memory on small hosts
        runs.pop(next(iter(runs)))
    runs[key] = run
    return run


# --------------------------------------------------------------------------- #
# Image helpers
# --------------------------------------------------------------------------- #
def to_rgb(img: np.ndarray, k: int = 3) -> np.ndarray:
    g = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    g = np.repeat(np.repeat(g, k, axis=0), k, axis=1)
    return np.stack([g, g, g], axis=-1)


def overlay(img: np.ndarray, cam: np.ndarray, k: int = 3) -> np.ndarray:
    base = to_rgb(img, k).astype(np.float32)
    a = np.repeat(np.repeat(np.clip(cam, 0, 1) * 0.7, k, axis=0), k, axis=1)[..., None]
    heat = np.array([235, 104, 52], dtype=np.float32)
    return (base * (1 - a) + heat * a).astype(np.uint8)


def gallery(images: list, captions: list, per_row: int = 8) -> None:
    for start in range(0, len(images), per_row):
        cols = st.columns(per_row)
        for col, im, cap in zip(cols, images[start:start + per_row], captions[start:start + per_row]):
            col.image(im, caption=cap, width="stretch")


def table(df: pd.DataFrame, **kw) -> None:
    st.dataframe(df, width="stretch", hide_index=True, **kw)


def md_table(df: pd.DataFrame) -> None:
    """Static table that wraps long text (st.dataframe truncates it)."""
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns)
    body = "\n".join("| " + " | ".join(str(v).replace("|", "/") for v in row) + " |" for row in df.itertuples(index=False))
    st.markdown(head + "\n" + body)


def plot(fig) -> None:
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


pct = lambda v: f"{v:.1%}"

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
M = tf_modules()
DEFAULT_M = M.ModelConfig()
st.session_state.setdefault("dcfg", DEFAULT_D)
st.session_state.setdefault("mcfg", DEFAULT_M)
FILTERS = {"8 - 16 - 32 (small)": (8, 16, 32), "16 - 32 - 64 (default)": (16, 32, 64), "32 - 64 - 96 (large)": (32, 64, 96)}

with st.sidebar:
    st.header("Decision settings")
    st.caption("These apply instantly. They change how the model's output is used, not the model.")
    threshold = st.slider("Decision threshold on P(defective)", 0.05, 0.95, 0.50, 0.01,
                          help="A part is flagged as defective when the CNN's probability is at or above this value.")
    cost_fp = st.number_input("Cost of a false alarm (good part flagged)", 0.0, 1000.0, 1.0, 1.0,
                              help="Re-inspection or scrapping of a good part.")
    cost_fn = st.number_input("Cost of a missed defect (bad part passed)", 0.0, 1000.0, 10.0, 1.0,
                              help="A defective part reaching assembly or the customer.")
    st.divider()
    st.header("Experiment settings")
    st.caption("Change these and press the button to generate a new dataset and train a new CNN "
               "(about a minute on a free host).")
    with st.form("experiment"):
        d0, m0 = st.session_state["dcfg"], st.session_state["mcfg"]
        st.subheader("Dataset")
        n_images = st.slider("Images", 1000, 6000, d0.n_images, 500)
        defect_rate = st.slider("Share of defective parts", 0.05, 0.60, d0.defect_rate, 0.05)
        noise_sigma = st.slider("Sensor noise", 0.0, 0.20, d0.noise_sigma, 0.01)
        sev = st.slider("Defect severity range", 0.10, 1.00, (d0.severity_min, d0.severity_max), 0.05,
                        help="0.1 = barely visible, 1.0 = obvious.")
        dtypes = st.multiselect("Defect types", DEFECT_TYPES, default=list(d0.defect_types))
        seed = st.number_input("Random seed", 0, 9999, d0.seed, 1)
        st.subheader("CNN")
        f_label = st.selectbox("Filters per convolution block", list(FILTERS),
                               index=list(FILTERS.values()).index(m0.filters) if m0.filters in FILTERS.values() else 1)
        dense_units = st.select_slider("Dense units", [16, 32, 64, 128], m0.dense_units)
        dropout = st.slider("Dropout", 0.0, 0.6, m0.dropout, 0.05)
        lr = st.select_slider("Learning rate", [3e-4, 1e-3, 3e-3], m0.learning_rate, format_func=lambda v: f"{v:g}")
        epochs = st.slider("Maximum epochs", 5, 50, m0.epochs, 5)
        patience = st.slider("Early-stopping patience", 2, 15, m0.early_stopping_patience, 1)
        go_btn = st.form_submit_button("Generate data and train CNN", type="primary", width="stretch")
    if st.button("Back to the pre-trained baseline", width="stretch"):
        st.session_state["dcfg"], st.session_state["mcfg"] = DEFAULT_D, DEFAULT_M
        st.rerun()

if go_btn:
    if not dtypes:
        st.sidebar.error("Pick at least one defect type.")
    else:
        st.session_state["dcfg"] = replace(DEFAULT_D, n_images=int(n_images), defect_rate=float(defect_rate),
                                           noise_sigma=float(noise_sigma), severity_min=float(sev[0]),
                                           severity_max=float(sev[1]), defect_types=tuple(dtypes), seed=int(seed))
        st.session_state["mcfg"] = replace(DEFAULT_M, filters=FILTERS[f_label], dense_units=int(dense_units),
                                           dropout=float(dropout), learning_rate=float(lr), epochs=int(epochs),
                                           early_stopping_patience=int(patience), seed=int(seed))

dcfg, mcfg = st.session_state["dcfg"], st.session_state["mcfg"]

# --------------------------------------------------------------------------- #
# Compute
# --------------------------------------------------------------------------- #
st.title("Vision-Based Preventive Maintenance with Camera Data")
st.caption("Session IV lab exercise · Reasoning and Decision-Making under Uncertainty · a CNN classifies synthetic "
           "camera images of machine parts as **normal** or **defective**.")

data = get_data(dcfg)
run = get_run(dcfg, mcfg)
model, info = run["model"], run["info"]
y_test, proba, meta_test, X_test = data["y_test"], run["proba_test"], data["meta_test"], data["X_test"]
m = ev.metrics_at(y_test, proba, threshold)
rk = ev.ranking_metrics(y_test, proba)
ci = ev.bootstrap_ci(y_test, proba, threshold)
sweep = ev.threshold_sweep(y_test, proba, cost_fp, cost_fn)
best_cost = sweep.loc[sweep["cost_per_1000"].idxmin()]
best_f1 = sweep.loc[sweep["f1"].idxmax()]
h = info["history"]
be = info["best_epoch"] - 1
train_m = ev.metrics_at(data["y_train"], run["proba_train"], threshold)

st.info(f"**Model in use:** {run['source']} · {info['params']:,} parameters · trained on {info['n_train']:,} images, "
        f"validated on {info['n_val']:,}, tested on {len(y_test):,} held-out images · threshold {threshold:.2f}")

k = st.columns(5)
ci_txt = {r["metric"]: f"95% CI {r['ci_low']:.1%} – {r['ci_high']:.1%}" for _, r in ci.iterrows()}
k[0].metric("Accuracy", pct(m["accuracy"]), help=ci_txt["accuracy"])
k[1].metric("Precision", pct(m["precision"]), help=ci_txt["precision"])
k[2].metric("Recall", pct(m["recall"]), help=ci_txt["recall"])
k[3].metric("F1 score", f"{m['f1']:.3f}", help=ci_txt["f1"])
k[4].metric("ROC-AUC", f"{rk['roc_auc']:.3f}", help="Threshold-free: 0.5 = random, 1.0 = perfect.")

tabs = st.tabs(["Overview", "1 · Dataset", "2 · Model & training", "3 · Evaluation", "4 · Threshold & cost",
                "5 · Error analysis", "6 · Stress test", "7 · Line monitor", "Interpretation"])

# ------------------------------------------------------------------ Overview
with tabs[0]:
    a, b = st.columns([3, 2])
    with a:
        st.subheader("Scenario")
        st.markdown(
            "A factory uses cameras to inspect parts for surface defects. This dashboard builds the vision model "
            "end to end: it **generates** synthetic camera images, **trains** a convolutional neural network, "
            "**evaluates** it on images it has never seen, and shows how the result would be **used on the line** "
            "to trigger maintenance before defects pile up.")
        st.subheader("What the test set says")
        st.markdown(
            f"- Of **{m['TP'] + m['FP']}** parts the model flagged, **{m['TP']}** were really defective "
            f"(precision {pct(m['precision'])}), so **{m['FP']}** were false alarms.\n"
            f"- Of **{m['TP'] + m['FN']}** really defective parts, it caught **{m['TP']}** (recall {pct(m['recall'])}) "
            f"and **missed {m['FN']}**.\n"
            f"- With a missed defect costing {cost_fn:g} and a false alarm {cost_fp:g}, the cheapest threshold is "
            f"**{best_cost['threshold']:.2f}** ({best_cost['cost_per_1000']:.0f} cost units per 1,000 parts, against "
            f"{sweep.loc[(sweep['threshold'] - threshold).abs().idxmin(), 'cost_per_1000']:.0f} at the current "
            f"{threshold:.2f}).")
    with b:
        st.subheader("Exercise checklist")
        md_table(pd.DataFrame([
            ("Synthetic images from noise and shape distortions", "1 · Dataset"),
            ("Two classes: normal and defective", "1 · Dataset"),
            ("CNN with convolution, pooling and dense layers", "2 · Model & training"),
            ("Binary classification output", "2 · Model & training"),
            ("Accuracy, precision, recall, F1", "3 · Evaluation"),
            ("Sample images from each class", "1 · Dataset"),
            ("Confusion matrix", "3 · Evaluation"),
            ("Training curves", "2 · Model & training"),
            ("Beyond the brief: threshold and cost, errors, stress test, SPC", "tabs 4 – 7"),
        ], columns=["Required by the exercise", "Where"]))
    st.subheader("Pipeline")
    st.markdown("`synthetic camera images` → `train / validation / test split` → `standardise pixels` → "
                "`CNN: 3 × (convolution + max-pooling) → dense → sigmoid` → `P(defective)` → `threshold` → "
                "`pass / reject` → `p-chart of reject rate` → `maintenance alert`")

# ------------------------------------------------------------------ Dataset
with tabs[1]:
    meta_all = pd.concat([data["meta_train"].assign(split="train + validation"), meta_test.assign(split="test")])
    c = st.columns(4)
    c[0].metric("Images", f"{len(meta_all):,}")
    c[1].metric("Normal", f"{int((meta_all.label == 0).sum()):,}")
    c[2].metric("Defective", f"{int((meta_all.label == 1).sum()):,}")
    c[3].metric("Image size (greyscale)", f"{dcfg.image_size} × {dcfg.image_size} px")

    st.subheader("Sample images: normal")
    idx = np.where(y_test == 0)[0][:8]
    gallery([to_rgb(X_test[i, ..., 0]) for i in idx], ["normal"] * len(idx))
    st.caption("Normal parts still differ in position, size, ovality, brightness, lighting and noise, and some carry "
               "harmless tooling marks or dust specks. The model has to learn that none of this is a defect.")

    st.subheader("Sample images: defective, by defect type")
    for dt in dcfg.defect_types:
        sub = meta_test[meta_test["defect_type"] == dt].sort_values("severity")
        if sub.empty:
            continue
        pick = sub.iloc[np.linspace(0, len(sub) - 1, min(8, len(sub))).astype(int)]
        st.markdown(f"**{dt.capitalize()}** — {DEFECT_INFO[dt]}")
        gallery([to_rgb(X_test[i, ..., 0]) for i in pick.index], [f"severity {s:.2f}" for s in pick["severity"]])
    st.caption("Each row runs from the subtlest example on the left to the most obvious on the right.")

    a, b = st.columns(2)
    with a:
        st.subheader("How the images are built")
        md_table(pd.DataFrame([
            ("Noise", "Gaussian sensor noise", f"σ = {dcfg.noise_sigma:.2f} on a 0 – 1 pixel scale"),
            ("Noise", "Uneven lighting", "Brightness gradient in a random direction"),
            ("Noise", "Surface texture and focus blur", "Low-frequency texture, machining rings, slight blur"),
            ("Shape (benign)", "Position, radius, ovality", "Vary in every part, normal or defective"),
            ("Shape (defect)", "Notch, dent", "A chip or a flat spot changes the outline"),
            ("Surface (defect)", "Scratch, pit, crack", "Marks on the surface"),
            ("Distractors", "Tooling marks, dust specks", "Faint, harmless, in both classes"),
        ], columns=["Source", "What", "Detail"]))
    with b:
        st.subheader("Split and class balance")
        split = meta_all.groupby("split").agg(images=("label", "size"), defective=("label", "sum")).reset_index()
        split["normal"] = split["images"] - split["defective"]
        split["share defective"] = (split["defective"] / split["images"]).map(pct)
        table(split[["split", "images", "normal", "defective", "share defective"]])
        st.caption(f"The test set ({dcfg.test_size:.0%}) is set aside before any training and stratified by defect "
                   f"type. {mcfg.validation_split:.0%} of the remaining images are held out for validation during "
                   "training.")
        table(meta_all[meta_all.label == 1].groupby("defect_type").agg(
            images=("label", "size"), mean_severity=("severity", "mean")).round(2).reset_index())

# ------------------------------------------------------------------ Model & training
with tabs[2]:
    a, b = st.columns([2, 3])
    with a:
        st.subheader("Architecture")
        lt = M.layer_table(model)
        table(lt, height=38 + 35 * len(lt))
        st.caption("Convolution layers find local patterns (edges, spots). Max-pooling halves the resolution and "
                   "keeps the strongest response. Dense layers combine the features, and one sigmoid unit outputs "
                   "P(defective).")
        st.subheader("Training set-up")
        md_table(pd.DataFrame([
            ("Loss", "Binary cross-entropy"), ("Optimiser", f"Adam, learning rate {mcfg.learning_rate:g}"),
            ("Batch size", mcfg.batch_size), ("Maximum epochs", mcfg.epochs),
            ("Early stopping", f"stop after {mcfg.early_stopping_patience} epochs without lower validation loss; "
                               "restore the best weights"),
            ("Dropout", f"{mcfg.dropout:.0%} before the output layer"),
            ("Pixel standardisation", "training-set mean and standard deviation, stored in the model"),
            ("Epochs run", f"{info['epochs_run']} (best: epoch {info['best_epoch']})"),
            ("Training time", f"{info['train_seconds']} s on CPU"),
        ], columns=["Setting", "Value"]).astype(str))
    with b:
        st.subheader("Training curves")
        plot(charts.training_curve(h, "accuracy", "Accuracy per epoch", "Accuracy", info["best_epoch"]))
        plot(charts.training_curve(h, "loss", "Loss per epoch (binary cross-entropy)", "Loss", info["best_epoch"], "top right"))
        gap = h["accuracy"][be] - h["val_accuracy"][be]
        late_gap = h["accuracy"][-1] - h["val_accuracy"][-1]
        st.markdown(
            f"**Reading the curves.** At the best epoch ({info['best_epoch']}) training accuracy is "
            f"{pct(h['accuracy'][be])} and validation accuracy {pct(h['val_accuracy'][be])}, a gap of "
            f"{gap * 100:+.1f} points. "
            + ("Training stopped early because validation loss stopped improving; " if info["stopped_early"] else
               "Training ran for the full number of epochs; ")
            + f"the weights from epoch {info['best_epoch']} are the ones used. "
            + (f"By the last epoch the gap had grown to {late_gap * 100:+.1f} points, the first sign of "
               "over-fitting: the model keeps improving on images it has seen while validation stalls. "
               if late_gap > gap + 0.01 else "")
            + f"On the untouched test set accuracy is {pct(ev.metrics_at(y_test, proba, 0.5)['accuracy'])} at "
              "threshold 0.50, close to validation, so the model generalises to new parts drawn the same way.")
        st.caption("In the lab demo the model reached 100% within one epoch because the synthetic classes were "
                   "trivially separable. Here defects vary in size and contrast and normal parts carry distractors, "
                   "so the curves take several epochs to rise and never reach 100%.")

# ------------------------------------------------------------------ Evaluation
with tabs[3]:
    a, b = st.columns([2, 3])
    with a:
        plot(charts.confusion_matrix(m))
    with b:
        st.subheader("Evaluation metrics on the test set")
        show = ci.assign(metric=ci["metric"].map({"accuracy": "Accuracy", "precision": "Precision",
                                                  "recall": "Recall", "f1": "F1 score"}))
        show["test set"] = show["value"].map(pct)
        show["95% confidence interval"] = [f"{lo:.1%} – {hi:.1%}" for lo, hi in zip(show["ci_low"], show["ci_high"])]
        show["training set"] = [pct(train_m[k_]) for k_ in ("accuracy", "precision", "recall", "f1")]
        show["meaning"] = ["Share of all parts classified correctly",
                           "Of the parts flagged, the share that are really defective (few false alarms)",
                           "Of the really defective parts, the share caught (few missed defects)",
                           "Harmonic mean of precision and recall"]
        md_table(show[["metric", "test set", "95% confidence interval", "training set", "meaning"]])
        st.caption("Confidence intervals come from 1,000 bootstrap resamples of the test set: they show how far each "
                   "number could move on a different sample of parts.")
        st.latex(rf"\text{{Accuracy}}=\frac{{TP+TN}}{{N}}=\frac{{{m['TP']}+{m['TN']}}}{{{len(y_test)}}}={m['accuracy']:.3f}")
        st.latex(rf"\text{{Precision}}=\frac{{TP}}{{TP+FP}}=\frac{{{m['TP']}}}{{{m['TP']}+{m['FP']}}}={m['precision']:.3f}")
        st.latex(rf"\text{{Recall}}=\frac{{TP}}{{TP+FN}}=\frac{{{m['TP']}}}{{{m['TP']}+{m['FN']}}}={m['recall']:.3f}")
        st.latex(rf"F_1=\frac{{2\cdot P\cdot R}}{{P+R}}={m['f1']:.3f}")
    c1, c2 = st.columns(2)
    cv = ev.curves(y_test, proba)
    with c1:
        plot(charts.roc(cv, rk["roc_auc"], m))
    with c2:
        plot(charts.precision_recall(cv, rk["pr_auc"], m, float(y_test.mean())))
    c1, c2 = st.columns(2)
    with c1:
        plot(charts.score_histogram(proba, y_test, threshold))
        st.caption("Blue bars on the right of the line are false alarms; orange bars on the left are missed defects.")
    with c2:
        plot(charts.calibration(ev.calibration_table(y_test, proba)))
        st.caption(f"Brier score {rk['brier']:.3f} (0 is perfect). Points near the diagonal mean the probability can "
                   "be read at face value when deciding how much to trust a single prediction.")

    st.subheader("Is a CNN needed? Comparison with two baselines")
    base = get_baselines(dcfg)
    rows, cnn_pred = [], (proba >= threshold).astype(int)
    for name, p_ in {"CNN (this model)": proba, **base}.items():
        bm = ev.metrics_at(y_test, p_, threshold if name.startswith("CNN") else 0.5)
        auc = ev.ranking_metrics(y_test, p_)["roc_auc"] if np.ptp(p_) > 0 else 0.5
        rows.append({"model": name, "accuracy": pct(bm["accuracy"]), "precision": pct(bm["precision"]),
                     "recall": pct(bm["recall"]), "F1": f"{bm['f1']:.3f}", "ROC-AUC": f"{auc:.3f}"})
    table(pd.DataFrame(rows))
    mc = ev.mcnemar(y_test, cnn_pred, (base["Logistic regression on pixels"] >= 0.5).astype(int))
    st.caption(f"Always answering 'normal' already scores {pct(1 - y_test.mean())} accuracy with zero recall, which is "
               "why accuracy alone is not enough. Logistic regression sees each pixel separately and cannot find a "
               "defect that moves around the part. McNemar's test on the parts where the two models disagree: the CNN "
               f"alone is right on {mc['only_a_correct']}, logistic regression alone on {mc['only_b_correct']} "
               f"(p = {mc['p_value']:.2g}).")

# ------------------------------------------------------------------ Threshold & cost
with tabs[4]:
    st.markdown("The CNN outputs a probability. The **threshold** turns it into a pass / reject decision, and moving "
                "it trades false alarms against missed defects. The lecture's rule: favour **precision** when "
                "unnecessary maintenance is expensive, favour **recall** when a missed failure is. Use the sidebar "
                "to move the threshold and set the two costs.")
    c1, c2 = st.columns(2)
    with c1:
        plot(charts.threshold_metrics(sweep, threshold))
    with c2:
        plot(charts.cost_curve(sweep, threshold, float(best_cost["threshold"])))
    cur = sweep.loc[(sweep["threshold"] - threshold).abs().idxmin()]
    rows = []
    for name, r in (("Current", cur), ("Highest F1", best_f1), ("Lowest expected cost", best_cost)):
        rows.append({"choice": name, "threshold": f"{r['threshold']:.2f}", "precision": pct(r["precision"]),
                     "recall": pct(r["recall"]), "F1": f"{r['f1']:.3f}", "false alarms": int(r["FP"]),
                     "missed defects": int(r["FN"]), "cost per 1,000 parts": f"{r['cost_per_1000']:.0f}"})
    table(pd.DataFrame(rows))
    saving = cur["cost_per_1000"] - best_cost["cost_per_1000"]
    st.markdown(
        f"With a missed defect costing **{cost_fn / max(cost_fp, 1e-9):.0f}×** a false alarm, the cheapest setting is a "
        f"threshold of **{best_cost['threshold']:.2f}**: it accepts {int(best_cost['FP'])} false alarms to cut missed "
        f"defects to {int(best_cost['FN'])}. "
        + (f"That is {saving:.0f} cost units per 1,000 parts cheaper than the current threshold."
           if saving > 0.5 else "The current threshold is already at or near the cheapest point."))
    st.caption("Caveat: the threshold is tuned on the same test set it is reported on, so the saving is slightly "
               "optimistic. On a real line it should be chosen on validation data and confirmed on fresh parts.")

# ------------------------------------------------------------------ Error analysis
with tabs[5]:
    c1, c2 = st.columns(2)
    with c1:
        pdr = ev.per_defect_recall(meta_test, proba, threshold)
        plot(charts.horizontal_bars(pdr, "defect_type", "recall", "Recall by defect type", "Share caught", "n"))
    with c2:
        rs = ev.recall_by_severity(meta_test, proba, threshold)
        if len(rs):
            plot(charts.stress_lines(rs, "severity", "Recall by defect severity (test set)", "Defect severity",
                                     cols=(("recall", "Recall", charts.ORANGE),)))
    if len(pdr):
        worst, bestd = pdr.iloc[0], pdr.iloc[-1]
        st.markdown(f"The model is weakest on **{worst['defect_type']}** defects ({pct(worst['recall'])} caught) and "
                    f"strongest on **{bestd['defect_type']}** ({pct(bestd['recall'])}). Misses concentrate at low "
                    "severity: small, low-contrast defects are the ones that get through.")

    pred = proba >= threshold
    fn_idx = np.where((y_test == 1) & ~pred)[0]
    fp_idx = np.where((y_test == 0) & pred)[0]
    tp_idx = np.where((y_test == 1) & pred)[0]
    fn_idx = fn_idx[np.argsort(proba[fn_idx])][:8]
    fp_idx = fp_idx[np.argsort(-proba[fp_idx])][:8]
    tp_idx = tp_idx[np.argsort(-proba[tp_idx])][:8]

    st.subheader(f"Missed defects (false negatives): {m['FN']} in the test set")
    if len(fn_idx):
        gallery([to_rgb(X_test[i, ..., 0]) for i in fn_idx],
                [f"{meta_test.defect_type[i]} · sev {meta_test.severity[i]:.2f} · P = {proba[i]:.2f}" for i in fn_idx])
    else:
        st.write("None at this threshold.")
    st.subheader(f"False alarms (false positives): {m['FP']} in the test set")
    if len(fp_idx):
        cams = M.grad_cam(model, X_test[fp_idx])
        gallery([overlay(X_test[i, ..., 0], cams[j]) for j, i in enumerate(fp_idx)],
                [f"normal · P = {proba[i]:.2f}" for i in fp_idx])
        st.caption("Orange shows the regions that pushed the score towards 'defective' (Grad-CAM). These parts have "
                   "no defect, so whatever is highlighted is a harmless feature the network mistook for one.")
    else:
        st.write("None at this threshold.")
    st.subheader("Correct detections, with the evidence the model used")
    if len(tp_idx):
        cams = M.grad_cam(model, X_test[tp_idx])
        gallery([to_rgb(X_test[i, ..., 0]) for i in tp_idx],
                [f"{meta_test.defect_type[i]} · P = {proba[i]:.2f}" for i in tp_idx])
        gallery([overlay(X_test[i, ..., 0], cams[j]) for j, i in enumerate(tp_idx)], ["Grad-CAM"] * len(tp_idx))
        st.caption("Top row: the camera image. Bottom row: the same image with the Grad-CAM heat map. Orange on the "
                   "defect means the model is right for the right reason; orange elsewhere means the score leans on "
                   "something other than the defect. The map is coarse (16 × 16 cells), so read it as a region, not "
                   "an outline.")

# ------------------------------------------------------------------ Stress test
with tabs[6]:
    st.markdown("The lecture's warning: a model that scores well on clean synthetic data may fail on real, noisier "
                "data. Each point below is a **fresh set of images the model has never seen**, generated under "
                "conditions that move away from the training conditions, and scored at the current threshold.")
    stress = get_stress(dcfg, mcfg, model)

    def frame(key: str) -> pd.DataFrame:
        rows = []
        for s in stress[key]:
            mm = ev.metrics_at(s["y"], s["proba"], threshold)
            rows.append({"value": s["value"], "accuracy": mm["accuracy"], "precision": mm["precision"],
                         "recall": mm["recall"], "flagged": (mm["TP"] + mm["FP"]) / len(s["y"])})
        return pd.DataFrame(rows)
    nf, lf, sf = frame("noise"), frame("lighting"), frame("severity")
    c1, c2 = st.columns(2)
    with c1:
        plot(charts.stress_lines(nf, "value", "More sensor noise than in training", "Sensor noise (σ)",
                                 trained_at=dcfg.noise_sigma))
    with c2:
        plot(charts.stress_lines(lf, "value", "Stronger uneven lighting than in training", "Lighting gradient strength",
                                 trained_at=dcfg.lighting))
    plot(charts.stress_lines(sf, "value", "Recall as defects get smaller and fainter", "Defect severity (fixed)",
                             cols=(("recall", "Recall", charts.ORANGE),)))
    at_train = nf.loc[(nf["value"] - dcfg.noise_sigma).abs().idxmin()]
    noisy = nf.iloc[-1]
    faint = sf.iloc[0]
    broken = nf[(nf["value"] > dcfg.noise_sigma) & (nf["accuracy"] < at_train["accuracy"] - 0.10)]
    if len(broken):
        b0 = broken.iloc[0]
        noise_txt = (f"accuracy is {pct(at_train['accuracy'])} at the training level (σ = {at_train['value']:.2f}) and "
                     f"falls to {pct(b0['accuracy'])} at σ = {b0['value']:.2f}. ")
        if noisy["flagged"] > 0.9:
            noise_txt += (f"At σ = {noisy['value']:.2f} the model flags {pct(noisy['flagged'])} of all parts: recall "
                          f"reads {pct(noisy['recall'])}, but precision is {pct(noisy['precision'])}, no better than "
                          "flagging everything. Recall on its own would hide this failure. ")
        elif noisy["flagged"] < 0.1:
            noise_txt += (f"At σ = {noisy['value']:.2f} the model flags only {pct(noisy['flagged'])} of parts and "
                          f"recall is {pct(noisy['recall'])}: it passes almost everything. ")
        noise_txt += ("The model learned what a clean image looks like and treats extra noise as surface damage, so a "
                      "dirtier lens or a cheaper sensor would require re-training on noisier images.")
    else:
        noise_txt = (f"accuracy is {pct(at_train['accuracy'])} at the training level and {pct(noisy['accuracy'])} at "
                     f"σ = {noisy['value']:.2f}; the model tolerates the noise range tested.")
    st.markdown(
        f"- **Noise:** {noise_txt}\n"
        f"- **Lighting:** accuracy goes from {pct(lf.iloc[0]['accuracy'])} with even light to "
        f"{pct(lf.iloc[-1]['accuracy'])} with the strongest gradient; precision moves from "
        f"{pct(lf.iloc[0]['precision'])} to {pct(lf.iloc[-1]['precision'])}.\n"
        f"- **Severity:** at severity {faint['value']:.1f} the model catches {pct(faint['recall'])} of defects; at 1.0 "
        f"it catches {pct(sf.iloc[-1]['recall'])}. The detection limit is a property of the camera resolution and "
        "noise as much as of the model.")

    rb = get_robust_noise(dcfg)
    if rb is not None and dcfg.image_size == rb["info"]["data_config"]["image_size"]:
        rcfg = rb["info"]["data_config"]
        st.subheader("Remedy: train on noisier data")
        st.markdown(
            "The lecture's answer to this risk is to *use more complex or noisy datasets for robustness*. The same "
            f"CNN architecture was trained a second time on {rcfg['n_images']:,} images in which every image draws "
            f"its own noise level between σ = {rcfg['noise_sigma']:.2f} and σ = {rcfg['noise_max']:.2f}. Both models "
            "are scored below on the same fresh images.")
        cmp = pd.DataFrame({"value": nf["value"], "current": nf["accuracy"],
                            "robust": [ev.metrics_at(s_["y"], s_["proba"], threshold)["accuracy"] for s_ in rb["noise"]]})
        c1, c2 = st.columns([3, 2])
        with c1:
            plot(charts.stress_lines(cmp, "value", "Accuracy as noise increases", "Sensor noise (σ)",
                                     trained_at=dcfg.noise_sigma,
                                     cols=(("current", "Model in use", charts.BLUE),
                                           ("robust", "Trained on varying noise", charts.ORANGE))))
        with c2:
            shown = cmp.rename(columns={"value": "Sensor noise (σ)", "current": "Model in use",
                                        "robust": "Trained on varying noise"}).copy()
            shown["Sensor noise (σ)"] = shown["Sensor noise (σ)"].map(lambda v: f"{v:.2f}")
            for col in ("Model in use", "Trained on varying noise"):
                shown[col] = shown[col].map(pct)
            table(shown)
        r_train = cmp.loc[(cmp["value"] - dcfg.noise_sigma).abs().idxmin()]
        r_mid = cmp.loc[(cmp["value"] - 0.08).abs().idxmin()]
        r_far = cmp.iloc[-1]
        st.markdown(
            f"- At the training noise level (σ = {r_train['value']:.2f}) the model in use scores "
            f"{pct(r_train['current'])} and the second model {pct(r_train['robust'])}"
            + ((f": robustness costs {(r_train['current'] - r_train['robust']) * 100:.1f} points of accuracy under "
                "clean conditions" + (", even with twice the training data.\n" if rcfg["n_images"] > dcfg.n_images
                                      else ".\n"))
               if r_train["current"] > r_train["robust"] else ".\n")
            + f"- At σ = {r_mid['value']:.2f} the second model holds {pct(r_mid['robust'])} while the model in use "
            f"is at {pct(r_mid['current'])}.\n"
            f"- At σ = {r_far['value']:.2f}, outside the range either model was trained on, the second model also "
            f"drops to {pct(r_far['robust'])}. Training on varied data protects only against the conditions it covers.")

# ------------------------------------------------------------------ Line monitor
with tabs[7]:
    st.subheader("Inspect one part")
    c = st.columns([1, 1, 1, 1])
    kind = c[0].selectbox("Part", ["normal", *dcfg.defect_types])
    sev1 = c[1].slider("Severity (if defective)", 0.1, 1.0, 0.6, 0.05)
    noise1 = c[2].slider("Sensor noise for this image", 0.0, 0.20, dcfg.noise_sigma, 0.01)
    if c[3].button("Take a new image", width="stretch"):
        st.session_state["shot"] = st.session_state.get("shot", 0) + 1
    rng = np.random.default_rng(1000 + st.session_state.get("shot", 0))
    img, _ = render_part(rng, replace(dcfg, noise_sigma=noise1), None if kind == "normal" else kind, sev1)
    p1 = float(M.predict_proba(model, img[None, ..., None])[0])
    cam1 = M.grad_cam(model, img[None, ..., None])[0]
    i1, i2, i3 = st.columns([1, 1, 2])
    i1.image(to_rgb(img, 4), caption="Camera image", width="stretch")
    i2.image(overlay(img, cam1 * p1, 4), caption="Evidence for 'defective' (fades when the score is low)",
             width="stretch")
    with i3:
        st.metric("P(defective)", f"{p1:.2f}")
        verdict = p1 >= threshold
        truth = kind != "normal"
        if verdict:
            st.error(f"🛑 REJECT — flagged as defective (threshold {threshold:.2f})")
        else:
            st.success(f"✅ PASS — classified as normal (threshold {threshold:.2f})")
        st.write("Correct." if verdict == truth else
                 ("Missed defect (false negative)." if truth else "False alarm (false positive)."))

    st.divider()
    st.subheader("From inspection to preventive maintenance: a control chart on the reject rate")
    st.markdown("One rejected part is a quality event. A **rising share** of rejected parts is a maintenance signal: "
                "the tool is wearing. The simulation below runs batches through the CNN, with a healthy process at "
                "first and tool wear setting in later, and plots the share flagged on a p-chart (statistical process "
                "control). Control limits come from the first batches only.")
    c = st.columns(5)
    n_batches = c[0].slider("Batches", 20, 50, 30, 5)
    batch_size = c[1].select_slider("Parts per batch", [50, 100, 200], 100)
    base_rate = c[2].slider("Healthy defect rate", 0.01, 0.10, 0.04, 0.01)
    wear_start = c[3].slider("Wear starts at batch", 8, n_batches - 2, min(19, n_batches - 2), 1)
    final_rate = c[4].slider("Defect rate at the end", 0.05, 0.40, 0.20, 0.01)
    baseline_batches = min(12, wear_start - 1)
    line = get_line(dcfg, mcfg, model, n_batches, batch_size, base_rate, wear_start, final_rate)
    flagged = (line["proba"] >= threshold).sum(axis=1)
    pc = ev.p_chart(flagged, batch_size, baseline_batches)
    plot(charts.spc_chart(pc, line["actual_rate"]))
    alarms = pc[pc["out_of_control"] & (pc["batch"] > baseline_batches)]
    false_alarm_batches = alarms[alarms["batch"] < wear_start]
    late = alarms[alarms["batch"] >= wear_start]
    c = st.columns(4)
    c[0].metric("Centre line (baseline reject rate)", pct(pc["centre"].iloc[0]))
    c[1].metric("Upper control limit", pct(pc["ucl"].iloc[0]))
    c[2].metric("First alarm", f"batch {int(late['batch'].iloc[0])}" if len(late) else "none")
    c[3].metric("Batches from wear to alarm", int(late["batch"].iloc[0]) - wear_start if len(late) else "–")
    if len(late):
        st.error(f"🛑 Maintenance alert at batch {int(late['batch'].iloc[0])}: the share flagged "
                 f"({pct(late['flagged_rate'].iloc[0])}) is above the upper control limit. Wear began at batch "
                 f"{wear_start}, so the chart reacted {int(late['batch'].iloc[0]) - wear_start} batch(es) later.")
    else:
        st.warning("⚠️ No point crossed the control limits. The wear is too small, or the batches too small, for this "
                   "chart to see it.")
    if len(false_alarm_batches):
        st.warning(f"⚠️ {len(false_alarm_batches)} alarm(s) before wear began (batch "
                   f"{', '.join(str(int(x)) for x in false_alarm_batches['batch'])}): a false alarm of the chart itself.")
    exp_flag = base_rate * m["recall"] + (1 - base_rate) * m["false_alarm_rate"]
    st.caption(f"The blue line is not the true defect rate. The CNN misses some defects and flags some good parts, so "
               f"a healthy process with {pct(base_rate)} real defects shows about {pct(exp_flag)} flagged "
               f"(= {pct(base_rate)} × recall {pct(m['recall'])} + {pct(1 - base_rate)} × false-alarm rate "
               f"{pct(m['false_alarm_rate'])}). The chart still works because it watches for a change. When a point "
               "leaves the limits, look for the root cause: tool wear, machine calibration, raw-material "
               "inconsistency, operator error or environmental change. A dirty lens also raises the flagged share, so "
               "check the camera too.")

# ------------------------------------------------------------------ Interpretation
with tabs[8]:
    st.subheader("Discussion and interpretation of the outcome")
    st.markdown(f"""
**1. Does the model work?** On {len(y_test):,} held-out images it classifies {pct(m['accuracy'])} correctly, with
ROC-AUC {rk['roc_auc']:.3f}. The majority-class baseline scores {pct(1 - y_test.mean())} and pixel-wise logistic regression
does little better, so the convolution layers are doing real work: they find a defect wherever it sits on the part.

**2. Precision or recall?** Precision is {pct(m['precision'])} ({m['FP']} false alarms) and recall {pct(m['recall'])}
({m['FN']} missed defects) at threshold {threshold:.2f}. In inspection a missed defect usually costs far more than a
second look at a good part, so recall is the number to protect. At the stated costs the cheapest threshold is
{best_cost['threshold']:.2f}, not the default 0.50.

**3. Is it over-fitted?** The lab demo hit 100% in one epoch, a sign of data that is too easy. Here training accuracy at
the best epoch is {pct(h['accuracy'][be])} against {pct(h['val_accuracy'][be])} validation and
{pct(ev.metrics_at(y_test, proba, 0.5)['accuracy'])} test. So there is a gap of {(h['accuracy'][be] - h['val_accuracy'][be]) * 100:.1f} points between training and validation:
mild over-fitting, held in check by dropout and early stopping (weights taken from epoch {info['best_epoch']} of
{info['epochs_run']} run). The stress test shows the larger risk: the model is only reliable under the conditions it
was trained on, and breaks with more noise or fainter defects.

**4. What does it miss?** Errors are not random. They sit in specific defect types and at low severity
(Error analysis tab). That tells the plant what to fix: better lighting or resolution for those defects, or more
training examples of them.

**5. How does it prevent failures?** A single reject removes one bad part. The control chart on the reject rate turns
many inspections into an early warning of tool wear, so maintenance can be scheduled before the defect rate becomes
a line stoppage.
""")
    st.subheader("Limits of this exercise")
    st.markdown("""
- **Synthetic data.** The images are drawn by a formula. Real parts have reflections, oil, varied backgrounds and defect
  types nobody listed. The numbers here show the method works; they are not a forecast of real-line performance.
- **Same generator for train and test.** The test set is unseen but comes from the same recipe. The stress test is the
  closest substitute for real out-of-distribution data.
- **One test set.** Confidence intervals show sampling error, but there is no cross-validation across training runs;
  a different seed gives slightly different results.
- **Class balance.** The dataset holds far more defective parts than a real line produces. With 1 – 5% defects,
  precision at the same threshold would be much lower, which is why the threshold should be set on the real mix.
""")
    st.subheader("Next steps")
    st.markdown("""
The lecture's three, applied here:
- **Test on unseen, real-world data.** Collect real camera images and fine-tune on them.
- **Add regularisation.** Dropout and early stopping are already in; cross-validation would show run-to-run spread.
- **Use more complex or noisier datasets.** The Stress test tab does this for noise and shows the trade-off: a model
  trained across noise levels survives a noisier camera but gives up some accuracy under clean conditions. The same
  should be done for lighting.

Beyond the lecture:
- Keep a person in the loop for parts scoring near the threshold, and feed their decisions back as training labels.
""")
    with st.expander("Reproducibility and disclosure"):
        st.json({"data_config": dcfg.to_dict(), "model_config": mcfg.to_dict(),
                 "training": {k_: v for k_, v in info.items() if k_ != "history"}})
        st.caption("GenAI disclosure: the code and text of this dashboard were generated with Claude (Anthropic) "
                   "and reviewed by the student. See the README for the prompt.")
