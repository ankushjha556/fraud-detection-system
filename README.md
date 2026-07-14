# 🛡️ Advanced Fraud Detection System

**A research-grade, multi-paradigm fraud detection pipeline** combining supervised, unsupervised, relational, and sequential learning — with fully documented methodology, including the failures and limitations most portfolios leave out.

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit-FF4B4B?style=for-the-badge)](#) &nbsp;
[![Report](https://img.shields.io/badge/📄_Full_Report-PDF-blue?style=for-the-badge)](docs/PROJECT_REPORT.md)

![Python](https://img.shields.io/badge/Python-3.11-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.11-EE4C2C) ![XGBoost](https://img.shields.io/badge/XGBoost-3.3-green) ![PyG](https://img.shields.io/badge/PyTorch_Geometric-2.8-orange) ![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 🎯 What This Is

Most student fraud-detection projects stop at "trained XGBoost, got 90% accuracy." This project asks a harder question: **does combining fundamentally different modeling paradigms — supervised trees, unsupervised representation learning, graph neural networks, and sequence models — produce genuinely better fraud detection, and what happens when it doesn't?**

Built on the [Kaggle Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (284,807 transactions, 492 frauds, 0.172% fraud rate).

## 🧠 Model Architecture

| Component | Technique | Purpose |
|---|---|---|
| **Baseline** | Logistic Regression | Establish the performance floor |
| **Core Classifier** | XGBoost + Optuna tuning | Best single-model precision/recall balance |
| **Representation Learning** | Variational Autoencoder | Unsupervised anomaly scoring on legitimate-only training |
| **Relational Learning** | GraphSAGE (GNN) | Fraud signal from feature-similarity transaction graphs |
| **Temporal Modeling** | Bidirectional LSTM | Velocity-pattern detection across transaction sequences |
| **Ensembling** | Stacked meta-learner (5-fold OOF) | Combines all signals via logistic regression |
| **Explainability** | SHAP + Integrated Gradients | Feature attribution for tree and neural components |
| **Uncertainty** | Monte Carlo Dropout | Flags low-confidence predictions for human review |
| **Monitoring** | Population Stability Index | Concept drift detection across time |

## 📊 Results

| Model | Precision | Recall | PR-AUC |
|---|---|---|---|
| Logistic Regression (baseline) | 0.920 | 0.622 | 0.749 |
| **XGBoost (Optuna-tuned)** | **0.827** | **0.838** | **0.837** |
| VAE (unsupervised) | — | — | 0.324 |
| LSTM (temporal) | 0.374 | 0.784 | 0.696 |
| GraphSAGE* | 0.722 | 0.878 | 0.902 |
| **Stacked Ensemble** (clean eval) | 0.966 | 0.781 | 0.807 |

*GraphSAGE evaluated on a rebalanced subsample — see report for why this isn't directly comparable.

**Full methodology, all caveats, and the complete ablation study:** [`docs/PROJECT_REPORT.md`](docs/PROJECT_REPORT.md)

## 🔬 What Makes This Different

- **Caught and fixed a real data leakage bug** during ensemble construction (66% of a "clean" evaluation set had leaked from training) — documented transparently rather than hidden
- **Reports when things don't work**: the stacked ensemble did *not* statistically beat XGBoost alone on PR-AUC — and the report explains why that's still a meaningful finding (see meta-learner coefficient analysis)
- **Honest about every approximation**: the GNN's graph is a similarity proxy (no real entity IDs exist in this dataset), the LSTM's "sequences" are time-windowed rather than per-entity, and MC Dropout uncertainty uses a repeated-transaction approximation — all explicitly flagged, not glossed over

## 🚀 Running Locally

```bash
git clone https://github.com/ankushjha556/fraud-detection-system.git
cd fraud-detection-system
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

The app runs with **real trained models** — XGBoost, VAE, LSTM, and the stacked meta-learner are all bundled directly in `models/` (total size under 1MB) and load automatically. The sidebar shows live per-component status (🟢/🔴) so it's transparent which models are active.

## 📓 Reproducing the Training Pipeline

All 9 stages are in [`notebooks/`](notebooks/), designed to run sequentially on Google Colab (T4 GPU recommended):

1. `01_setup_eda.ipynb` — Environment setup, EDA, shared train/val/test splits
2. `02_baseline_xgboost.ipynb` — Logistic regression baseline + tuned XGBoost + SHAP
3. `03_vae.ipynb` — Variational Autoencoder training + anomaly scoring
4. `04_gnn.ipynb` — Graph construction + GraphSAGE
5. `05_lstm.ipynb` — Sequence generation + LSTM training
6. `06_ensemble_explainability.ipynb` — Leakage-checked stacking + Integrated Gradients + MC Dropout
7. `07_drift_report.ipynb` — Concept drift simulation + final report generation

## 🗂️ Repo Structure

```
fraud-detection-system/
├── app/                    # Streamlit application
├── notebooks/               # Full training pipeline (Colab-ready)
├── models/                  # Trained artifacts (gitignored — see Releases)
├── docs/
│   └── PROJECT_REPORT.md    # Complete methodology + honest findings
├── assets/                  # Screenshots
└── requirements.txt
```

## 👤 Author

**Ankush Kumar Jha** — B.S. Computer Science & Data Analytics, IIT Patna
[GitHub](https://github.com/ankushjha556) · [LinkedIn](https://linkedin.com/in/ankush-jha-032696318)

## 📄 License

MIT
