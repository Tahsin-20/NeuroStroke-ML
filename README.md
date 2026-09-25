# Stroke Risk Prediction — Explainable ML

An end-to-end machine learning project that predicts stroke risk from demographic, lifestyle, and health data, with a focus on handling severe class imbalance responsibly and making the final model's predictions explainable rather than a black box.

🔗 **Live app:** *https://stroke-made-me-broke-byth.streamlit.app/*

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Handling Class Imbalance](#handling-class-imbalance)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Explainability (SHAP)](#explainability-shap)
- [Error Analysis](#error-analysis)
- [Deployment](#deployment)
- [Repository Structure](#repository-structure)
- [Running Locally](#running-locally)
- [Limitations & Disclaimer](#limitations--disclaimer)
- [Possible Improvements](#possible-improvements)

---

## Overview

Stroke is a leading cause of long-term disability, and early risk flagging from routine health data could support earlier intervention. This project walks through the full lifecycle of a health-risk classifier:

1. Exploratory data analysis and statistical feature-target association tests
2. A leak-free preprocessing pipeline (imputation, outlier capping, encoding, scaling)
3. Training and comparing four classification algorithms
4. Evaluating with imbalance-aware metrics rather than raw accuracy
5. Explaining the chosen model's predictions with SHAP
6. Auditing the model's mistakes directly
7. Deploying the final pipeline as an interactive Streamlit app

## Dataset

[Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) — ~5,000 patient records with demographic (age, gender), lifestyle (work type, residence, smoking status, marital status), and health (hypertension, heart disease, average glucose level, BMI) features, with a binary `stroke` target.

The target is highly imbalanced — roughly **5% of patients had a stroke** — which shapes almost every modeling decision below.

## Project Workflow

| Stage | What's covered |
|---|---|
| EDA | Missing values, duplicates, distributions, skewness, outliers, target imbalance |
| Statistical tests | Point-biserial correlation (numeric vs. target), Cramér's V (categorical vs. target) |
| Preprocessing | Custom IQR-capping transformer, median imputation, standard scaling, ordinal + one-hot encoding — all inside a single `sklearn` `Pipeline` to avoid data leakage |
| Modeling | Logistic Regression, Decision Tree, KNN, and a small Neural Network (MLP), compared under identical stratified cross-validation |
| Evaluation | Accuracy, precision, recall, F1, ROC-AUC, PR-AUC, PR curves, ROC curves, confusion matrix |
| Explainability | SHAP global and local explanations for the selected model |
| Error analysis | Manual inspection of false negatives / false positives |
| Deployment | Streamlit app serving the saved pipeline for live predictions |

## Handling Class Imbalance

Because stroke cases are rare, the project deliberately avoids accuracy as the primary success metric — a model that always predicts "no stroke" would already be ~95% accurate while being clinically useless. Instead:

- Stratified train/test split and stratified k-fold cross-validation preserve the true class ratio in every split.
- Models are trained with `class_weight="balanced"` so the rare positive class isn't ignored.
- Precision, recall, F1, ROC-AUC, and **PR-AUC** (more informative than ROC-AUC under heavy imbalance) are tracked together, not accuracy alone.

## Modeling

All four models share the same preprocessing pipeline for a fair comparison:

- **Logistic Regression** — final selected model
- **Decision Tree** (depth-limited)
- **K-Nearest Neighbors**
- **Neural Network** (small MLP)

Logistic Regression was selected as the final model for its balance of solid recall on the minority class, calibrated probability outputs, and — importantly — interpretability, which pairs naturally with SHAP's exact linear explainer.

## Evaluation

Models are compared via:
- 5-fold stratified cross-validation on the training set
- A held-out test set evaluated once, at the end
- Precision-recall curves and ROC curves across all four models
- A confusion matrix for the final model

Full numeric results and plots are in the notebook — see `Stroke_Prediction.ipynb` below.

## Explainability (SHAP)

The final model's predictions are explained using [SHAP](https://github.com/shap/shap):

- **Global importance** — a beeswarm summary plot and a mean-|SHAP| bar chart showing which features drive predictions overall (age and average glucose level dominate).
- **Local explanations** — waterfall plots that break down one individual prediction feature-by-feature, including a walkthrough of a false negative to see what masked that patient's risk.
- **Dependence plots** — how a feature's value relates to its effect on the prediction across the whole test set.

This turns the model from "here's a risk score" into "here's *why* this patient got this score," which matters for any health-adjacent prediction.

## Error Analysis

Misclassified patients are pulled out and inspected directly rather than left inside a single confusion-matrix number, to get a feel for which patient profiles the model struggles with — particularly the false negatives, which are the costlier error type in a stroke-screening context.

## Deployment

The trained pipeline (`models/stroke_pipeline.pkl`) is served through a [Streamlit](https://streamlit.io/) app, where a user can enter a patient's details and get a live risk prediction.


<img width="557" height="809" alt="Screenshot 2026-09-25 at 3 03 40 PM" src="https://github.com/user-attachments/assets/2b5b9029-7103-431a-a789-8e58a0ee1783" />



## Repository Structure

```
.
├── Stroke_Prediction.ipynb      # Full analysis: EDA → modeling → SHAP
├── models/
│   └── stroke_pipeline.pkl          # Saved final pipeline (preprocessing + model)
├── app.py                           # Streamlit app
├── requirements.txt
├── .gitignore
├── healthcare-dataset-stroke-data.csv
└── README.md
```

## Running Locally

```bash
# 1. Clone the repo
git clone <https://github.com/Tahsin-20/NeuroStroke-ML>
cd <NeuroStroke-ML>

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

**Core dependencies:** `pandas`, `numpy`, `scikit-learn`, `shap`, `matplotlib`, `seaborn`, `scipy`, `joblib`, `streamlit`

## Limitations & Disclaimer

- This model is trained on a single public dataset and has **not** been clinically validated. It is a portfolio/learning project, not a diagnostic tool, and should not be used to inform real medical decisions.
- The dataset's positive class is small (~5%), so performance on rare subgroups within that class may be unstable.
- No external test set or deployment-time monitoring is in place — real-world data drift is not accounted for.

## Possible Improvements

- Hyperparameter tuning (e.g. `GridSearchCV` / `Optuna`) rather than hand-picked defaults
- Threshold tuning tied to an explicit cost assumption (missed strokes vs. false alarms)
- A gradient-boosted tree model (XGBoost/LightGBM) as an additional comparison point, paired with SHAP's `TreeExplainer`
- Probability calibration check (calibration curve / Brier score)
- Surfacing a per-prediction SHAP explanation directly in the Streamlit app
