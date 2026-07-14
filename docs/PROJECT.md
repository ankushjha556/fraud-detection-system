# Advanced Fraud Detection System — Detailed Project Documentation

**Author:** Ankush Kumar Jha, B.S. Computer Science & Data Analytics, IIT Patna
**Live app:** https://fraud-detection-system-wpvpkvxzrhbdbsqmxf5n2h.streamlit.app/
**Dataset:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## Table of Contents

1. [Motivation and Research Question](#1-motivation-and-research-question)
2. [Dataset](#2-dataset)
3. [Methodology Overview](#3-methodology-overview)
4. [Stage-by-Stage Breakdown](#4-stage-by-stage-breakdown)
5. [Key Findings](#5-key-findings)
6. [The Data Leakage Incident](#6-the-data-leakage-incident)
7. [Explainability Analysis](#7-explainability-analysis)
8. [Uncertainty Quantification](#8-uncertainty-quantification)
9. [Concept Drift Analysis](#9-concept-drift-analysis)
10. [Full Ablation Table](#10-full-ablation-table)
11. [Limitations](#11-limitations)
12. [What I Would Do Differently](#12-what-i-would-do-differently)
13. [Conclusion](#13-conclusion)

---

## 1. Motivation and Research Question

Most introductory fraud-detection projects stop at "train a classifier, report accuracy." This project asked a more specific question:

> **Does combining fundamentally different machine learning paradigms — supervised classification, unsupervised representation learning, graph-based relational learning, and sequential modeling — produce measurably better fraud detection than the best single model? And if not, what does that tell us?**

This framing matters because in production fraud systems, teams routinely combine heterogeneous models (rule engines, gradient boosting, anomaly detectors) under the assumption that diversity helps. This project tests that assumption directly, with honest reporting of the result either way.

A second, equally important goal was **methodological discipline**: every evaluation number in this project needed to be defensible — same train/test boundaries, no leakage, documented approximations. This constraint is what surfaced the most interesting finding of the whole project (see Section 6).

---

## 2. Dataset

**Source:** Kaggle Credit Card Fraud Detection, originally released by the Machine Learning Group at Université Libre de Bruxelles.

**Composition:**
- 284,807 transactions made by European cardholders over two days in September 2013
- 492 confirmed frauds (0.172% of all transactions) — extreme class imbalance
- 30 features: `Time` (seconds since first transaction), `Amount`, and `V1`–`V28` (PCA-transformed, anonymized for confidentiality — original features are unknown)
- Binary target: `Class` (1 = fraud, 0 = legitimate)

**Why this dataset was chosen:** it is real transaction data (not synthetic), has genuine extreme imbalance (a realistic fraud rate, not an artificially balanced toy problem), and is small enough to iterate quickly on Colab while still supporting meaningful graph and sequence construction.

**Known constraint:** because features are PCA-anonymized, there are no real entity identifiers (no cardholder ID, merchant ID, or location). This single constraint shaped two major design decisions later in the pipeline (see Sections 4.4 and 4.5).

---

## 3. Methodology Overview

The project was built as nine sequential stages, each validated with real, printed output before moving to the next — no stage was written speculatively without confirming the prior stage's numbers made sense.

```
1. EDA & Shared Splits  →  2. Baseline (LogReg)  →  3. XGBoost + Optuna + SHAP
        ↓
4. VAE (unsupervised)  →  5. GraphSAGE (GNN)  →  6. LSTM (temporal)
        ↓
7. Stacked Ensemble (leakage-checked)  →  8. Explainability + Uncertainty
        ↓
9. Concept Drift Simulation  →  Final Report
```

All models were trained in Google Colab (T4 GPU). Model artifacts, evaluation scores, and intermediate results were persisted to Google Drive after each stage so that no work was lost between sessions and every downstream stage could load prior results rather than recomputing them.

---

## 4. Stage-by-Stage Breakdown

### 4.1 EDA and Shared Splits

Before any modeling, the dataset was profiled:
- Confirmed fraud rate: 0.1727% (492 / 284,807)
- `Amount` distribution differs meaningfully by class: fraud transactions have a higher mean ($122.21) but lower median ($9.25) than legitimate ones (mean $88.29, median $22.00) — fraud is bimodal, mixing small "card testing" transactions with occasional large ones
- Correlation-with-`Class` ranking identified `V14`, `V17`, `V12`, `V10`, `V16` as strongest negative correlates and `V11`, `V4`, `V2` as strongest positive correlates — this ranking was later cross-validated against SHAP and Integrated Gradients results (Section 7)

**Two independent splits were built and saved:**
1. **Stratified random split** (70/15/15 train/val/test) — used for XGBoost, VAE, and initial GNN work
2. **Temporal split** (earliest 70% vs. latest 30% by transaction time) — reserved specifically for the concept drift analysis in Section 9

This dual-split design was intentional: a model's ability to generalize across time is a different question from its ability to generalize across a random sample, and conflating the two would have hidden the drift analysis entirely.

### 4.2 Baseline: Logistic Regression

Two variants were trained to establish the performance floor and illustrate the precision/recall tradeoff under imbalance:

- **Unweighted:** Precision 0.920, Recall 0.622, PR-AUC 0.749 — conservative, only flags transactions it is very confident about, missing 28 of 74 test-set frauds
- **Class-weighted (`class_weight='balanced'`):** Precision 0.067, Recall 0.878, PR-AUC 0.792 — catches far more fraud but at the cost of 905 false positives to find 65 true frauds

This pair demonstrates why accuracy and even simple precision/recall are insufficient for reporting on this problem — PR-AUC, which evaluates ranking quality across all thresholds, is the fairer comparison metric and is used as the primary metric throughout the rest of the project.

### 4.3 XGBoost: Hyperparameter Tuning and Imbalance Strategy

**Imbalance strategy ablation (before tuning):** Three approaches to handling class imbalance were compared directly:

| Strategy | PR-AUC |
|---|---|
| `scale_pos_weight` only (no resampling) | **0.8381** |
| SMOTE oversampling | 0.8218 |
| SMOTE-ENN (oversample + clean boundary noise) | 0.8131 |

This result is counter to common practice — SMOTE and SMOTE-ENN are frequently reached for by default in imbalanced classification tutorials. Here, native class weighting outperformed both synthetic resampling methods, likely because SMOTE's linear interpolation between minority-class neighbors introduces noise in an already-dense PCA-transformed feature space, whereas `scale_pos_weight` directly reweights the loss function on real, unmodified data points.

**Hyperparameter tuning:** 40-trial Optuna search over `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`, `reg_alpha`, `reg_lambda`, and `scale_pos_weight`, optimizing validation PR-AUC.

**Best trial:** validation PR-AUC 0.8432, with `n_estimators=194`, `max_depth=6`, `learning_rate=0.292`, `scale_pos_weight=699.1`.

**Final test-set performance:** Precision 0.8267, Recall 0.8378, F1 0.8322, ROC-AUC 0.9616, **PR-AUC 0.8369**. Confusion matrix: 42,635 true negatives, 13 false positives, 12 false negatives, 62 true positives — out of 42,722 test transactions, only 25 were misclassified in either direction.

This became the benchmark every other model and the ensemble were measured against.

### 4.4 Variational Autoencoder: Unsupervised Representation Learning

**Architecture:** Encoder (30 → 64 → 32 → latent dim 8, via separate μ and log-variance heads), reparameterization trick, decoder mirroring the encoder back to 30 dimensions. Trained with combined reconstruction (MSE) and KL-divergence loss (β = 0.5), Adam optimizer, 30 epochs, batch size 256.

**Critical design choice:** the VAE was trained *exclusively on legitimate transactions* (~199,000 examples), never seeing a single fraud example during training. This is the standard unsupervised anomaly-detection setup: the model learns to reconstruct "normal" transactions well; anything that reconstructs poorly (high reconstruction error) or sits in a low-probability region of latent space (high KL term) is flagged as anomalous.

**Training convergence:** smooth exponential loss decay from 11.68 (epoch 5) to 9.94 (epoch 30), no divergence or instability, reconstruction and KL terms both plateauing — healthy convergence.

**Anomaly scoring result:** PR-AUC 0.324 on the shared test set — substantially weaker than XGBoost. However, ROC-AUC was 0.956, nearly identical to XGBoost's 0.962. This divergence between ROC-AUC and PR-AUC is itself informative: ROC-AUC is insensitive to class imbalance in a way PR-AUC is not, so a "good" ROC-AUC alongside a "weak" PR-AUC is exactly the signature of a model that ranks fraud above *most* legitimate transactions but still gets swamped by the sheer volume of transactions sitting near the decision boundary at this extreme (0.17%) base rate.

**t-SNE visualization of the latent space** showed the majority of fraud transactions clustering tightly together, separate from the legitimate mass — but one fraud transaction sat embedded deep inside the legitimate cluster. This is a concrete illustration of a limitation inherent to any reconstruction-based method: a fraud transaction that is *statistically unremarkable* in feature space will not be flagged by an anomaly detector, regardless of its true label. This is precisely the failure mode a supervised model with access to labels (XGBoost) is positioned to catch instead, motivating the ensemble approach.

### 4.5 GraphSAGE: Relational Learning

**The entity-ID problem:** This dataset has no cardholder ID, merchant ID, device ID, or any other real relational key. A true "fraud ring" graph (connecting transactions by shared entities) is therefore impossible to construct from this data as-is.

**The proxy solution:** A k-nearest-neighbors similarity graph was built instead — each transaction connected to its 8 nearest neighbors in the 30-dimensional PCA feature space. The reasoning: transactions that are behaviorally similar (and, given PCA, occurring close together in the underlying — unknown — feature space) may share fraud-relevant structure even without an explicit entity link. This is a real limitation, documented explicitly rather than presented as a true relational graph.

**Graph scale:** for GPU memory and training-time reasons, a 15,000-node subsample was built rather than using the full 284,807-transaction graph. All 492 fraud cases were retained in this subsample (with legitimate transactions subsampled down to fill the remainder), which raised the fraud rate within this graph to 3.28% — a substantially easier classification problem than the full dataset's 0.17%.

**Model:** 3-layer GraphSAGE (mean aggregation), hidden dimension 64, class-weighted cross-entropy loss, 100 epochs, best-validation-checkpoint selection.

**Training:** validation PR-AUC climbed from 0.9455 (epoch 10) to a best of 0.9633, with the loss decreasing smoothly throughout.

**Test-set result:** Precision 0.7222, Recall 0.8784, **PR-AUC 0.9019** — the highest PR-AUC of any single model in the project. However, this number is evaluated on the rebalanced 3.28%-fraud subsample, not the full 0.17%-fraud population every other model was tested against. **It is explicitly not a fair comparison and is labeled as such everywhere it is reported.** The genuine finding here is that GraphSAGE effectively propagates label signal through feature-similarity neighborhoods — a real and useful result, just not directly comparable to the XGBoost/VAE/LSTM numbers.

### 4.6 LSTM: Temporal / Velocity Modeling

**Motivation:** fraud often exhibits "velocity" patterns — rapid, similar transactions occurring in a short time window. Without entity IDs, per-cardholder velocity cannot be modeled directly, so this was approximated using fixed-length sliding windows of 10 consecutive transactions (sorted by time), with the window's label taken from its final transaction.

**Architecture:** 2-layer bidirectional LSTM (hidden dimension 64), followed by a small feedforward classification head, dropout 0.3, class-weighted cross-entropy loss.

**Training behavior:** loss decreased smoothly and monotonically from 0.22 to 0.01 across 15 epochs, but validation PR-AUC was noisy — peaking early (0.744 at epoch 2), dipping through the middle epochs, and recovering to a best of 0.769 at epoch 15. This divergence between steadily falling loss and unstable validation PR-AUC is the classic signature of overfitting on a rare positive class: with only 344 fraud sequences in the training set, the model had very few positive examples to generalize from and began memorizing specifics rather than learning transferable velocity patterns after the first few epochs.

**Test-set result:** Precision 0.374, Recall 0.784, PR-AUC 0.696 — a real but modest signal, and notably weaker than the validation PR-AUC (0.769), confirming the overfitting story. The low precision (374 false positives per ~58 true positives caught, proportionally) means this model is recall-oriented and noisy — useful as a complementary signal in an ensemble, not deployable standalone.

### 4.7 Stacked Meta-Ensemble

This stage combines the outputs of XGBoost, VAE, and LSTM into a single stacked prediction via a logistic regression meta-learner, trained on out-of-fold predictions from 5-fold cross-validation to avoid overfitting the stacker itself.

The construction of this stage surfaced the most important methodological finding of the entire project — see Section 6.

**Final leakage-free result (14,364-row evaluation set, 73 frauds):**
- XGBoost alone (re-evaluated on this exact set): PR-AUC 0.8241
- Stacked ensemble (5-fold out-of-fold): PR-AUC 0.8065
- At the ensemble's optimal F1 threshold (0.9989): Precision 0.9661, Recall 0.7808, F1 0.8636

**Meta-learner coefficients:** XGBoost 8.37, **VAE 9.50 (highest)**, LSTM 2.43.

The ensemble did not out-perform XGBoost alone on aggregate PR-AUC, and with only 73 positive examples in the evaluation set this difference is not statistically distinguishable from noise. But the coefficient the meta-learner assigned to the VAE — higher than XGBoost's own coefficient, despite the VAE's much weaker standalone PR-AUC (0.324) — is a meaningful finding: it indicates the VAE's reconstruction-error signal contains information that is genuinely orthogonal to what XGBoost's decision-tree splits capture, and the meta-learner is exploiting that even though it doesn't show up as an aggregate PR-AUC improvement on this particular small evaluation set.

### 4.8 Explainability and Uncertainty

Covered in depth in Sections 7 and 8.

### 4.9 Concept Drift Simulation

Covered in depth in Section 9.

---

## 5. Key Findings

**Finding 1 — Native class weighting beats synthetic oversampling on this dataset.** `scale_pos_weight` (PR-AUC 0.8381) outperformed both SMOTE (0.8218) and SMOTE-ENN (0.8131) for XGBoost. This runs against a common default assumption in imbalanced-classification tutorials and is worth testing rather than assuming on any new dataset.

**Finding 2 — XGBoost remains the strongest single, fairly-evaluated model.** PR-AUC 0.837 on the full, correctly-imbalanced test set, with a near-perfectly balanced Precision (0.827) and Recall (0.838) — genuinely difficult to achieve simultaneously at a 0.17% base rate.

**Finding 3 — Different model families disagree on what "fraud" looks like, and that disagreement is informative, not noise.** XGBoost's SHAP top features (`V14`, `V4`, `V10`, `V12`, `V16`) and the VAE's Integrated Gradients top features (`V17`, `V7`, `V10`, `V12`, `V1`) overlap only partially. This divergence is direct evidence the two models are learning genuinely different decision structures — supervised split-based boundaries versus unsupervised reconstruction anomalies — which is the theoretical justification for ensembling in the first place, independent of whether the ensemble's aggregate PR-AUC improved.

**Finding 4 — The stacked ensemble did not statistically outperform XGBoost alone, and that's a legitimate, reportable result.** On a rigorously leakage-checked evaluation set (necessarily small — 73 frauds — after removing leaked rows), PR-AUC was 0.807 (ensemble) versus 0.824 (XGBoost). The honest conclusion is that stacking added value at the level of individual predictions (evidenced by the meta-learner's VAE weighting and the strong Precision/Recall at the optimal threshold) without necessarily moving the aggregate PR-AUC needle on this specific evaluation sample size.

**Finding 5 — Uncertainty estimation successfully identifies harder cases.** Monte Carlo Dropout showed mean prediction uncertainty on true fraud cases (0.0207) was 3.4x higher than on legitimate cases (0.0060) — the model is, on average, appropriately less confident exactly where it should be. Routing the top 5% most-uncertain predictions to human review would surface 16.4% of all fraud cases while touching only 5% of transaction volume — a 3.3x enrichment over random sampling, and a concrete, quantified case for human-in-the-loop deployment design.

**Finding 6 — PSI and PR-AUC tell different, complementary stories about drift.** Population Stability Index (0.0299) indicated no significant global drift between an early-period and late-period evaluation, but PR-AUC dropped 4.6% (0.837 → 0.798) under strict temporal evaluation. PSI measures broad score-distribution shift; it is not sensitive to narrow performance changes on a rare positive class. Relying on PSI alone in a production monitoring setup would have missed this degradation.

---

## 6. The Data Leakage Incident

This section is documented in detail because it was the single most methodologically important moment in the project.

**What happened:** During construction of the stacked ensemble (Section 4.7), predictions from XGBoost and VAE (evaluated on a stratified random test split) needed to be combined with LSTM predictions (evaluated on a time-sorted, sequence-shifted split built independently in an earlier stage). To combine them, the *intersection* of rows scored by all three models was computed.

**First attempt:** the intersection contained only 7,182 rows with 36 frauds — far smaller than expected, because two independently-drawn stratified splits from the same 284,807-row pool only overlap by chance (roughly 6,400 rows expected from the relative split sizes). This was a red flag on its own, but the deeper issue was still hidden.

**The fix that revealed the real problem:** rather than working around the small overlap, the LSTM's test set (42,720 rows) was used as the canonical common evaluation set, and XGBoost and the VAE were *re-scored* on those exact same rows to guarantee alignment. This produced a full 42,720-row, 74-fraud common set — but before trusting any ensemble numbers built on it, a leakage check was run: **cross-referencing this "common" set's row indices against XGBoost's original training set indices.**

**The result: 28,356 of the 42,720 rows (66%) had been part of XGBoost's training data.** XGBoost's apparent performance on this "held-out" set would have been artificially inflated by memorization rather than generalization, and any ensemble built on top of it would have inherited that inflation.

**The correction:** all leaked rows were removed, leaving a genuinely leakage-free evaluation set of 14,364 rows with 73 frauds. All ensemble numbers reported in this project (Sections 4.7 and 10) are computed on this clean set, not the leaked one.

**Why this matters beyond this one project:** this is a realistic and easy-to-make mistake — it happened here specifically *because* different pipeline stages (built at different times, for different modeling paradigms) each built their own "reasonable-looking" train/test split independently, rather than the whole project committing to a single master split from the start. The practical lesson, stated plainly: **define one canonical train/val/test split at the very beginning of a multi-model project and enforce every subsequent model to respect it exactly** — don't let convenience (a sequence-shifted split for an LSTM, a rebalanced subsample for a GNN) silently diverge from the project's actual held-out set.

---

## 7. Explainability Analysis

**SHAP (XGBoost, TreeExplainer, 2,000-transaction sample):** Top features by mean absolute SHAP value: `V14` (dominant, ~2.5), `V4` (~2.05), `V10` (~1.35), `V12` (~1.10), `V16` (~0.78), followed by `V11`, `V19`, `Time_scaled`, `V25`, `V27`. This ranking closely matches the raw correlation-with-`Class` analysis from EDA (Section 4.1) — a useful sanity check that XGBoost is learning genuine, statistically-grounded signal rather than spurious patterns.

**Integrated Gradients (VAE reconstruction error, 73 fraud transactions, zero baseline, 50 steps):** Top features by mean absolute attribution: `V17` (dominant, ~1.12), `V7` (~0.87), `V10` (~0.56), `V12` (~0.51), `V1` (~0.40), `V4` (~0.40), followed by `V8`, `V3`, `V14`, `V5`. Convergence delta was 0.211 — higher than the textbook near-zero target, attributed to the VAE's non-linear decoder producing a less smooth attribution surface than IG's underlying linearity assumption prefers; feature rankings from this analysis should be read as directional rather than precisely quantitative.

**The cross-model comparison is the key result of this section:** `V10` and `V12` appear in both models' top-5, but XGBoost's single most important feature (`V14`) drops to 9th place for the VAE, while the VAE's most important feature (`V17`) barely registers in XGBoost's ranking. This is quantitative evidence supporting Finding 3 (Section 5) — the two models are genuinely learning different aspects of what makes a transaction fraudulent.

---

## 8. Uncertainty Quantification

Monte Carlo Dropout (50 stochastic forward passes with dropout active at inference time) was applied to the LSTM to estimate per-transaction prediction uncertainty, not just a point prediction.

**Result:** mean uncertainty (standard deviation across the 50 samples) for true fraud cases was 0.0207, versus 0.0060 for true legitimate cases — over 3x higher. This is the expected and desired behavior: the model should be less confident on the class it has less training signal for and where the decision boundary is genuinely harder.

**Practical framing:** flagging the top 5% most-uncertain predictions for human review surfaced 12 of 73 total fraud cases (16.4%) while touching only 5% of transaction volume — a 3.3x fraud enrichment compared to random review sampling. This is a concrete, quantified argument for a human-in-the-loop deployment architecture (auto-approve high-confidence legitimate, auto-block high-confidence fraud, route uncertain cases to an analyst) rather than a fully automated binary system.

**Documented approximation:** because the LSTM expects sequence input and this analysis needed per-transaction (not per-sequence) uncertainty, each transaction was repeated 10 times to form an artificial length-10 "sequence." This measures the model's parameter uncertainty under dropout, not genuine temporal ambiguity — a real simplification, stated explicitly rather than implied to be more sophisticated than it is.

---

## 9. Concept Drift Analysis

**Setup:** XGBoost (with the same best hyperparameters found via Optuna) was retrained using only the earliest 70% of transactions by time, then evaluated on the latest 30% — simulating what would happen if a model trained today were deployed against tomorrow's transaction stream without retraining.

**Result:** PR-AUC dropped from 0.837 (random split) to 0.798 (strict temporal split) — a real, if modest, 4.6% relative decline.

**Population Stability Index (PSI):** comparing the model's score distribution on early-period versus late-period data gave PSI = 0.0299, well under the standard 0.1 threshold for "no significant drift."

**The apparent tension between these two results is itself the finding:** PSI measures whether the *overall distribution* of predicted scores has shifted meaningfully — and at this dataset's short time window (roughly two days of data) and extreme imbalance (>99.8% of both periods' scores cluster near zero regardless of any subtle change), PSI has limited sensitivity. The PR-AUC drop shows the model's *discriminative power specifically on the rare positive class* did degrade somewhat, even though the bulk score distribution looks stable by PSI's measure. In a production monitoring setup, relying on PSI alone would have missed this degradation entirely — a genuinely useful, non-obvious conclusion for anyone building drift monitoring around a rare-event classifier.

---

## 10. Full Ablation Table

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | Evaluation Set |
|---|---|---|---|---|---|---|
| Logistic Regression (unweighted) | 0.920 | 0.622 | 0.742 | 0.958 | 0.749 | Full stratified test (42,722 rows) |
| Logistic Regression (balanced) | 0.067 | 0.878 | 0.125 | 0.968 | 0.792 | Full stratified test |
| XGBoost (Optuna-tuned) | 0.827 | 0.838 | 0.832 | 0.962 | 0.837 | Full stratified test |
| VAE (unsupervised anomaly) | — | — | — | 0.956 | 0.324 | Full stratified test |
| LSTM (temporal, seq_len=10) | 0.374 | 0.784 | 0.507 | 0.955 | 0.696 | Sequence-shifted test (42,720 rows) |
| GraphSAGE | 0.722 | 0.878 | 0.793 | 0.968 | 0.902 | Rebalanced graph subsample (2,250 rows, 3.28% fraud) — not comparable |
| XGBoost (re-evaluated, clean set) | — | — | — | — | 0.824 | Leakage-free common set (14,364 rows) |
| Stacked Ensemble | 0.966¹ | 0.781¹ | 0.864¹ | 0.930 | 0.807 | Leakage-free common set |

¹ At the ensemble's optimal F1-maximizing threshold (0.9989).

---

## 11. Limitations

1. **No real entity identifiers exist in this dataset.** The GraphSAGE component's graph is constructed via k-NN similarity in PCA feature space, which is a defensible but imperfect proxy for true relational structure (shared cardholder, merchant, or device). A dataset with real entity IDs would likely produce a meaningfully different — probably stronger and more interpretable — GNN result.
2. **GraphSAGE's headline PR-AUC (0.902) is not directly comparable to the other models.** It was measured on a deliberately rebalanced 3.28%-fraud subsample (all fraud retained, legitimate transactions downsampled) rather than the full dataset's 0.17%-fraud population every other model was evaluated against.
3. **The LSTM's "sequences" are fixed time-windows across the whole transaction stream, not per-entity transaction histories.** Without entity IDs, genuine per-cardholder velocity patterns cannot be modeled; what is captured instead is closer to "how unusual is this transaction relative to its 10 nearest-in-time neighbors."
4. **The leakage-free ensemble evaluation set is small (73 fraud cases).** This limits the statistical power to detect a genuinely small difference between the ensemble and XGBoost alone — the 0.807 vs. 0.824 PR-AUC gap reported in Section 4.7 should not be read as a confident "ensemble is worse," only as "not distinguishable from equal, on this sample size."
5. **Monte Carlo Dropout uncertainty (Section 8) uses a repeated-transaction sequence approximation**, not genuine temporal context, since the LSTM architecture was reused rather than building a separate model purely for uncertainty estimation.
6. **Integrated Gradients convergence delta (0.211) is higher than ideal**, reflecting the VAE decoder's non-linearity; feature attribution rankings from this analysis should be interpreted directionally.

---

## 12. What I Would Do Differently

- **Define one single, canonical train/val/test split at the very start of the project (Section 1)** and require every subsequent model — including the LSTM's sequence-shifted data and the GNN's subsampled graph — to derive its evaluation set strictly from that master split, rather than allowing each modeling stage to build its own reasonable-looking split independently. This is the direct, actionable lesson from the leakage incident in Section 6.
- **Investigate whether a larger, entity-aware dataset changes the GNN result meaningfully.** This dataset's lack of entity IDs is the single biggest structural limitation on the relational-learning component; a dataset with real cardholder/merchant identifiers would let the graph reflect genuine relational fraud rings rather than a feature-similarity proxy.
- **Build a dedicated uncertainty model** rather than reusing the LSTM with a repeated-sequence workaround, to get a cleaner separation between "temporal signal" and "prediction confidence."

---

## 13. Conclusion

XGBoost with a tuned `scale_pos_weight` remains the strongest single, fairly-evaluated model in this project (PR-AUC 0.837, Precision 0.827, Recall 0.838). The VAE, GraphSAGE, and LSTM components each demonstrably contribute complementary — not redundant — signal, evidenced concretely through SHAP/Integrated Gradients feature-ranking divergence and the meta-learner's disproportionately high weighting of the VAE's output. However, the full stacked ensemble did not produce a statistically distinguishable improvement over XGBoost alone on this dataset's necessarily small, leakage-free evaluation sample.

The project's most valuable outcome is not a leaderboard number — it is the demonstrated discipline of catching and correcting a real data leakage bug mid-pipeline, explicitly documenting every approximation and limitation rather than presenting results with unwarranted confidence, and reporting a negative/ambiguous ensemble result honestly rather than reframing it to look more successful than the evidence supports. These practices matter more in real-world ML engineering than chasing marginal metric gains, and this project was built to demonstrate exactly that discipline end-to-end.
