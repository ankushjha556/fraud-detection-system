# 🛡️ Advanced Fraud Detection System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fraud-detection-system-wpvpkvxzrhbdbsqmxf5n2h.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11-EE4C2C)](https://pytorch.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.3-green)](https://xgboost.readthedocs.io)
[![PyTorch Geometric](https://img.shields.io/badge/PyTorch_Geometric-2.8-orange)](https://pytorch-geometric.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

**🚀 [Live Demo](https://fraud-detection-system-wpvpkvxzrhbdbsqmxf5n2h.streamlit.app/)** &nbsp;·&nbsp; **📄 [Detailed Project Documentation](docs/PROJECT.md)** &nbsp;·&nbsp;

---

## What This Is

A multi-paradigm fraud detection system built on the Kaggle Credit Card Fraud dataset (284,807 transactions, 492 frauds, 0.172% fraud rate). Five distinct modeling approaches — supervised gradient boosting, unsupervised representation learning, graph neural networks, sequence modeling, and stacked ensembling — are trained, evaluated, and honestly compared against each other.

The project's actual contribution isn't a single leaderboard number. It's a **fully documented, leakage-checked comparison** of what each paradigm adds (and doesn't add) to fraud detection, including a real data leakage bug that was caught mid-pipeline and fixed rather than silently absorbed into inflated metrics.

The [live app](https://fraud-detection-system-wpvpkvxzrhbdbsqmxf5n2h.streamlit.app/) runs the actual trained models (XGBoost, VAE, LSTM, and the meta-learner) for real-time transaction scoring — not a mock.

## Model Architecture

| Stage | Technique | Role |
|---|---|---|
| Baseline | Logistic Regression | Performance floor |
| Core classifier | XGBoost + Optuna (40-trial search) | Best single-model precision/recall balance |
| Representation learning | Variational Autoencoder | Unsupervised anomaly scoring, trained on legitimate transactions only |
| Relational learning | GraphSAGE (GNN) | Fraud signal from a k-NN transaction similarity graph |
| Temporal modeling | Bidirectional LSTM | Velocity-pattern detection across transaction sequences |
| Ensembling | Stacked meta-learner (5-fold out-of-fold logistic regression) | Combines all four signals |
| Explainability | SHAP (XGBoost) + Integrated Gradients (VAE) | Feature attribution across model types |
| Uncertainty | Monte Carlo Dropout | Flags low-confidence predictions for human review |
| Monitoring | Population Stability Index | Concept drift detection across time |

## Results

| Model | Precision | Recall | PR-AUC |
|---|---|---|---|
| Logistic Regression (unweighted) | 0.920 | 0.622 | 0.749 |
| Logistic Regression (balanced) | 0.067 | 0.878 | 0.792 |
| **XGBoost (Optuna-tuned)** | **0.827** | **0.838** | **0.837** |
| VAE (unsupervised) | — | — | 0.324 |
| LSTM (temporal) | 0.374 | 0.784 | 0.696 |
| GraphSAGE¹ | 0.722 | 0.878 | 0.902 |
| Stacked Ensemble² | 0.966 | 0.781 | 0.807 |

¹ Evaluated on a rebalanced 3.28%-fraud subsample — not directly comparable to the others (see report).
² Evaluated on a leakage-checked, held-out 14,364-row set (73 frauds) after a leakage bug was found and corrected.

XGBoost with tuned `scale_pos_weight` is the strongest single, directly-comparable model. The stacked ensemble did not statistically beat it on PR-AUC given the small clean evaluation set, but its meta-learner assigned the VAE the *highest* weight of the three base models — evidence the VAE captures complementary signal despite a weaker standalone score. Full reasoning in [docs/PROJECT.md](docs/PROJECT.md).

## What Makes This Different From a Typical Student Project

- **A real data leakage bug was found and fixed, not hidden.** 66% of an initial "clean" ensemble evaluation set had leaked from XGBoost's training data. This was caught by cross-checking indices across independently-built splits, and the evaluation set was rebuilt before any ensemble numbers were trusted.
- **Negative and ambiguous results are reported as such.** The ensemble not beating XGBoost, GraphSAGE's non-comparable evaluation sample, the LSTM's overfitting — all documented with explanation rather than cherry-picked around.
- **Every approximation is flagged.** No real entity IDs exist in this dataset, so the GNN's graph is a feature-similarity proxy and the LSTM's "sequences" are time-windowed rather than per-entity. MC Dropout uncertainty uses a repeated-transaction approximation. All stated plainly, not glossed over.

## Running Locally

```bash
git clone https://github.com/ankushjha556/fraud-detection-system.git
cd fraud-detection-system
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

All model artifacts (XGBoost, VAE, LSTM, meta-learner — under 1MB combined) are bundled in `models/` and load automatically. The sidebar shows live per-component status.

## Training Pipeline

The full 9-stage pipeline (EDA → baseline → XGBoost → VAE → GraphSAGE → LSTM → ensemble → explainability → drift analysis) was developed and run in Google Colab. Notebook exports will be added to [`notebooks/`](notebooks/) — see [docs/PROJECT.md](docs/PROJECT.md) for the complete stage-by-stage methodology, exact hyperparameters, and code-level details in the meantime.

## Repository Structure

```
fraud-detection-system/
├── app/
│   ├── streamlit_app.py       # Main application
│   └── models_def/
│       ├── architectures.py   # PyTorch model class definitions
│       └── inference.py       # Real-model loading and scoring
├── models/                    # Trained artifacts (XGBoost, VAE, LSTM, meta-learner)
├── notebooks/                 # Training pipeline notebooks (in progress)
├── docs/
│   ├── PROJECT.md             # Full detailed methodology, findings, limitations
│   └── PROJECT_DESCRIPTION.md 
├── requirements.txt
└── README.md
```

## Author

**Ankush Kumar Jha**
B.S. Computer Science & Data Analytics, IIT Patna
[GitHub](https://github.com/ankushjha556) · [LinkedIn](https://linkedin.com/in/ankush-jha-032696318)

## License

MIT
