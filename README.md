# Vision-Based Preventive Maintenance with Camera Data (Synthetic)

Session IV lab exercise, *Reasoning and Decision-Making under Uncertainty* (Day 6: Quality Control and Predictive Maintenance).

A factory camera inspects machine parts for surface defects. This project generates synthetic camera images, trains a
convolutional neural network (CNN) to classify each part as **normal** or **defective**, evaluates it on unseen images,
and presents everything in an interactive Streamlit dashboard.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

The dashboard opens with a pre-trained baseline CNN (`artifacts/baseline_model.keras`), so it loads in seconds. The
sidebar lets you change the dataset and CNN settings and train a new model in the browser (about a minute on CPU).

**Streamlit Community Cloud:** New app → this repository → branch `main` → main file `app.py`. Works on Python 3.12, 3.13 and 3.14 (on 3.14 the requirements install the TensorFlow 2.22 release candidate, because 2.21 has no 3.14 build).

## How the exercise is covered

| Exercise requirement | Where in the dashboard | Code |
|---|---|---|
| Synthetic images generated with noise and shape distortions | 1 · Dataset | `vision_pm/data.py` |
| Two classes: normal and defective | 1 · Dataset | `vision_pm/data.py` |
| CNN with convolution, pooling and dense layers | 2 · Model & training | `vision_pm/model.py` |
| Binary classification output (sigmoid) | 2 · Model & training | `vision_pm/model.py` |
| Accuracy, precision, recall, F1 score | header and 3 · Evaluation | `vision_pm/evaluate.py` |
| Sample images from each class | 1 · Dataset | `app.py` |
| Confusion matrix | 3 · Evaluation | `vision_pm/charts.py` |
| Training curves (accuracy and loss) | 2 · Model & training | `vision_pm/charts.py` |

Beyond the brief: bootstrap confidence intervals, ROC and precision-recall curves, calibration, comparison with two
baselines (majority class, logistic regression) and McNemar's test, cost-based threshold choice, error analysis with
Grad-CAM, a stress test under noisier and unevenly lit images, and a p-chart (SPC) that turns the reject rate into a
maintenance alert.

## Dataset

3,000 greyscale 64 × 64 images of a ring-shaped part, 40% defective, split 75 / 25 into train and test (stratified by
defect type); 20% of the training images are held out for validation.

- **Noise:** Gaussian sensor noise, uneven lighting, surface texture, slight blur.
- **Benign variation (both classes):** position, size, ovality, brightness, faint tooling marks, dust specks.
- **Defects:** notch and dent (shape distortions), scratch, pit and crack (surface marks), each with a severity from
  0.3 (subtle) to 1.0 (obvious).

The lab demo in the lecture reached 100% accuracy in one epoch because its two classes were trivially separable. This
dataset is built so that the task is learnable but not trivial.

## Model

`Input 64×64×1 → standardise → [Conv2D 3×3 + ReLU → MaxPooling 2×2] × 3 (16, 32, 64 filters) → Flatten → Dense 64 →
Dropout 0.3 → Dense 1 (sigmoid)`; 285,569 parameters; Adam, binary cross-entropy, up to 30 epochs, early stopping on
validation loss with the best weights restored.

## Results (baseline, 750 unseen test images, threshold 0.50)

| Metric | Value | 95% bootstrap CI |
|---|---|---|
| Accuracy | 92.4% | 90.5% – 94.1% |
| Precision | 93.2% | 90.2% – 95.9% |
| Recall | 87.3% | 83.7% – 90.8% |
| F1 score | 0.902 | 0.877 – 0.925 |
| ROC-AUC | 0.966 | |

Confusion matrix: 431 true negatives, 19 false alarms, 38 missed defects, 262 true positives.

What the analysis shows:

- **A CNN is needed.** Always answering "normal" scores 60.0%; logistic regression on pixels scores 63.7%.
- **The default threshold is not the cheapest.** If a missed defect costs 10× a false alarm, the lowest-cost threshold
  is 0.14 (recall 96.0%, precision 76.2%), not 0.50.
- **Errors are concentrated.** Cracks, notches and pits are caught 98 – 100% of the time; scratches 76% and dents 63%,
  mostly at low severity.
- **The model is brittle outside its training conditions.** Trained at noise σ = 0.05, accuracy falls from 91.5% to
  53.0% at σ = 0.07 on fresh images, and from σ = 0.08 the model flags almost every part. This is the lecture's warning about
  synthetic data.
- **Training on noisier data fixes most of it, at a price.** The same CNN trained on 6,000 images with noise varying
  from σ = 0.03 to 0.10 holds 87.0% at σ = 0.08 (against 42.5%), but scores 89.5% instead of 91.5% at σ = 0.05.
- **Mild over-fitting.** Training accuracy is 97.2% against 93.1% validation at the best epoch (24 of 30).

## Limits

- The images are synthetic. The results show that the method works; they do not predict accuracy on a real line.
- Train and test images come from the same generator. The stress test is the closest substitute for real unseen data.
- One train/test split and one training run per model; no cross-validation.
- 40% of the images are defective, far more than on a real line. The threshold should be set on the real mix.
- The cost-optimal threshold is chosen on the test set, so its saving is slightly optimistic.

## Repository

```
app.py                  Streamlit dashboard
train_baseline.py       Re-trains and saves the two shipped models
vision_pm/data.py       Synthetic image generator
vision_pm/model.py      CNN, training, prediction, Grad-CAM
vision_pm/evaluate.py   Metrics, confidence intervals, threshold sweep, baselines, p-chart
vision_pm/analysis.py   Stress test and production-line simulation
vision_pm/charts.py     Plotly charts
artifacts/              Saved models and their training records
tests/                  pytest suite (python -m pytest)
```

## GenAI disclosure

The code and text in this repository were generated with Claude (Anthropic) and reviewed by the student. The prompt is
in [PROMPT.md](PROMPT.md).
