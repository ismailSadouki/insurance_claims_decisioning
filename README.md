# Insurance Claims Decisioning

> A three-stage end-to-end machine learning study on a heavily-obfuscated insurance claims dataset (~113K rows, 150+ features). The project moves through **predicting** claim outcomes (supervised learning), **understanding** the geometry of the feature space (clustering + dimensionality reduction), and **deciding** what action to take on each claim (reinforcement learning shaped by the supervised model). The deliverables are three full analytical PDFs; this README is the entry point.

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Models](https://img.shields.io/badge/models-LGBM%20%7C%20CatBoost%20%7C%20XGBoost%20%7C%20H2O-blue.svg)](#stage-1--supervised-learning)
[![Reports](https://img.shields.io/badge/reports-3%20PDFs-blueviolet.svg)](#reports)

---

## Executive Summary

| | |
|---|---|
| **Dataset** | ~113K insurance claims, 150+ features (112 numerical, 19 categorical), 76.1% / 23.9% class split (3.17:1 imbalance) |
| **Stage 1 — Supervised** | Best model: **CatBoost** (ROC-AUC 0.7755, PR-AUC 0.9140, OOF pAUC 0.630) |
| **Stage 2 — Clustering/DR** | PCA: 4 components explain 80% variance. Clustering: all methods unanimously find **k=2**, driven by a single feature threshold (depth-1 tree, AUC=1.0). Split does **not** align with target (ARI ≈ 0) |
| **Stage 3 — RL** | Tabular Q-learning with reward shaping from supervised probability. Avg reward +2.39 (beats all baselines) but **riskier** than supervised (557 vs 24 risky fast-tracks) → hybrid deployment recommended |
| **Deliverables** | 3 PDFs (6.4 MB + 7.6 MB + 283 KB), 8+ Jupyter notebooks, custom Q-learning implementation |

---

## Reports

The deliverables for this project are three full analytical reports. Each is self-contained — readable without running any of the notebooks.

| # | Report | What it covers |
|---|--------|----------------|
| 1 | [`1. supervised learning.pdf`](./1.%20supervised%20learning.pdf) | Full EDA, missingness analysis, feature engineering (8 binary thresholds from CDF crossovers), 6 hyperparameter strategies, model benchmarking, error analysis, EDA-vs-error cross-reference, 5 final recommendations |
| 2 | [`2. Clustering and Dimensionality Reduction Analysis.pdf`](./2.%20Clustering%20and%20Dimensionality%20Reduction%20Analysis.pdf) | PCA / Kernel PCA / Factor Analysis / t-SNE / UMAP / NMF / autoencoder benchmark, correlation-cluster diagnosis, 6 clustering methods with bootstrap stability, sub-cluster analysis |
| 3 | [`3. RL.pdf`](./3.%20RL.pdf) | Full RL environment design (state / action / reward), Q-learning training, policy comparison, hybrid deployment recommendation |

---

## Key Analytical Insight: The Column Names Are Fictitious

The single most important finding from the EDA — and the foundation for every subsequent analytical decision — is that **the column names in this dataset do not correspond to their stated meaning**. This is provable statistically: several pairs of "unrelated" features show correlations that are mathematically impossible if the names were real.

| Feature pair | Correlation | Why impossible if names were real |
|---|---:|---|
| `witnesses_count` ↔ `api_response_quality` | r = 0.998 | A count and a quality score for different systems cannot be 99.8% correlated |
| `network_discount_rate` ↔ `renewal_count` | r = 0.994 | A rate and a count have different units and meaning |
| `claim_complexity_score` ↔ `liability_percentage` | r = 0.993 | A score and a percentage for different concepts |
| `claim_validity_score` ↔ `external_verifications_count` | r = 0.992 | Unrelated measurements |
| `customer_tenure_years` ↔ `line_items_count` | r = 0.981 | Age vs. count |
| `reinsurance_amount` ↔ `specialist_referrals` | r = 0.980 | Financial amount vs. referral count |

**Consequence:** Every claim and decision in this project is grounded in **statistical evidence only** — KS statistic, AUC direction, mutual information, correlation structure, t-tests, ANOVA, chi-square. Never in what a column name implies. The 11,175 pairwise correlations revealed that 5,053 pairs exceed |r| ≥ 0.85, and 7 correlation clusters at r ≥ 0.99 account for 107 of the 150 features. The dataset has only **~50 truly independent signals** — the rest are rescaled copies.

---

## Three-Stage Pipeline

```
                  Stage 1                       Stage 2                       Stage 3
              ┌─────────────────┐           ┌─────────────────┐           ┌─────────────────┐
              │   Supervised    │    ──►    │  Clustering +   │    ──►    │   Reinforcement │
              │   Learning      │           │  Dim. Reduction │           │   Learning      │
              └─────────────────┘           └─────────────────┘           └─────────────────┘
                     │                              │                              │
                     ▼                              ▼                              ▼
            Predict claim outcome      Understand feature-space         Decide action per claim
            (fraud vs. legitimate)     geometry & natural clusters     (approve / docs / review / deny)
                     │                              │                              │
                     └──────────── P(claim=1) feeds RL state & reward shaping ──────┘
```

The supervised model's predicted probability feeds into the RL agent's state and reward function, making the three stages a connected pipeline rather than three independent tasks.

---

## Stage 1 — Supervised Learning

**Task:** Binary classification on ~90.8K training / ~22.7K test insurance claims.

### EDA highlights

- **Missingness is MNAR, not MCAR.** Missingness is bimodal at the row level (rows are either ~0% or ~100% missing). A logistic regression trained purely on the binary missingness matrix (no feature values — just which columns are absent per row) scores **AUC = 0.60** on 5-fold CV. This proves the pattern of missingness carries predictive signal.
- **Hierarchical clustering of missingness patterns** → engineered features → AUC lift 0.7038 → 0.7089 (+0.0051). The engineered features rank among the top permutation-importance features in the final model.
- **CDF-crossover analysis** identified 8 optimal binary thresholds from the distributions where Class 0 vs Class 1 cumulative distributions diverge most:

  | Feature | Threshold | Engineered feature |
  |---|---:|---|
  | `prior_denial_indicator` | > 0 | `has_prior_denial` |
  | `customer_value_score` | > 0 | `has_customer_value` |
  | `communication_touchpoints` | == 0 | `is_zero_touchpoints` |
  | `coverage_limit` | > 12.5 | `high_coverage_limit` |
  | `policy_premium_annual` | > 8 | `high_premium` |
  | `claim_processing_days` | < 1 | `is_fast_claim` |
  | `recovery_probability` | == 1 | `recovery_is_one` |
  | `policy_renewal_months` | == 7 | `renewal_is_7` |

- **Statistical testing** at the univariate level: t-test/ANOVA F-statistics (top: `prior_denial_indicator` F=5,742), mutual information (top categorical: `claim_detail_code` MI=0.115, 18,210 unique values), chi-square (top: `claim_detail_code` χ²=18,343). 16 features with p > 0.05 flagged as drop candidates for linear models.

### Five modeling properties (from EDA → modeling decisions)

1. **Weak individual features, strong collective signal** — strongest single numerical feature has AUC = 0.687. → Deep trees + ensembles required.
2. **Non-monotone signal in key features** — `policy_renewal_months` has KS=0.080 but AUC=0.501 (signal is a spike at value 7, not a ranking). → Linear models will miss this entirely.
3. **Extreme cardinality in the most informative categorical** — `claim_detail_code` has 18,210 unique values and MI=0.115. → CatBoost's ordered target statistics are the natural fit.
4. **Fabricated column names, true latent structure** — 3,416 feature pairs with r > 0.97. → Deduplication is mandatory, not optional.
5. **Moderate class imbalance (3.17:1)** — All models need `class_weight` or per-sample weighting.

### Modeling & hyperparameter optimization

Six hyperparameter strategies were benchmarked, not just one:

| Strategy | Sampler / Method | Notes |
|---|---|---|
| Bayesian TPE (Optuna default) | Tree-structured Parzen Estimator | Baseline |
| Hyperopt | TPE (independent implementation) | Cross-validation of TPE |
| Population-Based Training (PBT) | Evolutionary refinement of parallel trials | Broader exploration |
| **NSGA-II + CMA-ES (multi-objective)** | Pareto front: maximize OOF pAUC + minimize train/val gap | Final selected strategy |
| Genetic Algorithms | DEAP-based | Similar results to TPE |
| Hyperband / Successive Halving | Resource-aware | Stopped early — no improvement |

The multi-objective Pareto front simultaneously maximized OOF pAUC and minimized the train/val gap. Three selection strategies defined on the Pareto front:
- **Strategy A**: best pAUC (ignore gap)
- **Strategy B**: best gap (most regularized)
- **Strategy C**: 70% pAUC + 30% gap weighted — **selected as default**

### Final model comparison

| Model | Tuning | OOF AUC | OOF pAUC | Test AUC | Test pAUC | Class-0 F1 |
|---|---|---:|---:|---:|---:|---:|
| **CatBoost** | Bayesian + PBT | — | **0.6303** | **0.7755** | **0.6349** | **0.5253** |
| LightGBM | Hyperopt TPE | 0.7586 | 0.6226 | 0.7563 | 0.6197 | 0.5076 |
| LightGBM | PBT | 0.7589 | 0.6221 | 0.7554 | 0.6188 | 0.5074 |
| LightGBM | Optuna TPE | 0.7563 | 0.6213 | 0.7529 | 0.6161 | 0.4920 |
| LightGBM | NSGA-II/CMA-ES | — | ~0.618 | 0.7498 | 0.6127 | 0.3944 |
| XGBoost | Bayesian + PBT | 0.7589 | 0.6220 | 0.7564 | 0.6196 | 0.5063 |
| Logistic Regression | — | — | — | ~0.7324 | — | — |
| SGD Classifier | — | — | — | ~0.7331 | — | — |
| LDA | — | — | — | ~0.7153 | — | — |
| H2O StackedEnsemble | AutoML | — | — | 0.7610 | — | — |

Linear baselines top out ~3% AUC lower than tree models, confirming the problem is non-linear. CatBoost wins because of its native handling of `claim_detail_code` via ordered target statistics.

### Error analysis (on best model)

| Metric | Value | Interpretation |
|---|---:|---|
| ROC-AUC | 0.7755 | Moderate; consistent with EDA prediction (no single feature > 0.70) |
| PR-AUC | 0.9140 | High but inflated by 76% class-1 prevalence (baseline = 0.761) |
| Brier Score | 0.1778 | Reasonable sharpness; calibration would lower this |
| F1 Class 0 | 0.5253 | Weak; minority class heavily underfit |
| F1 Class 1 | 0.7872 | Acceptable for majority class |
| False Positives | 2,614 | High-confidence errors (avg prob 0.66) |
| False Negatives | 7,463 | Low-confidence errors (avg prob 0.39); 3× more numerous than FPs |
| KS(FP vs FN) | 1.000 | FP and FN distributions perfectly separable |

The EDA-vs-error cross-reference table confirmed every EDA prediction on the test set: class imbalance → class-0 under-prediction (1.337× impact), `prior_denial_indicator` dominance → rank 2 SHAP, U-shaped `previous_claims_value` → U-shaped error rate, `policy_renewal_months` non-monotonicity → non-monotone error pattern, etc.

**Five final recommendations:** (1) Platt scaling / isotonic recalibration, (2) threshold re-optimization from PR curve (0.247 vs current 0.52), (3) deduplicate the 3,416 correlated feature pairs, (4) investigate distribution shift on 3 features, (5) target the FP-dominant cluster (n=285, 75.8% FP) with a post-processing rule.

📄 **Full report:** [`1. supervised learning.pdf`](./1.%20supervised%20learning.pdf)

---

## Stage 2 — Clustering & Dimensionality Reduction

**Task:** Understand the intrinsic geometry of the 150-feature space and test whether natural clusters align with the classification target.

### Dimensionality reduction findings

| Method | Key finding |
|---|---|
| **PCA** | **4 components explain 80% of variance; PC1 alone = 66.95%.** Reconstruction MSE near-zero at k=50. |
| Kernel PCA (5 kernels) | rbf / poly / sigmoid / cosine produce visually similar 2D class distributions to linear PCA → no strong non-linear structure |
| Factor Analysis | 10 factors; high communalities for operational metrics, zero communality for rule-based flags (retain as standalone predictors) |
| t-SNE (4 perplexities) | KL diverges 1.06 → 0.75 from p=5 to p=100; classes mixed at all perplexities |
| UMAP (12 configs) | Best visual separation; reveals ~20–30 subpopulations at standard (n=15, d=0.1) |
| NMF | Degenerate distribution — most points compressed along one axis |
| Autoencoder (5-layer MLP) | Best for subpopulation discovery; reconstruction MSE 0.298 vs PCA 0.228 (likely overfits at k=21) |

**Class separation:** No 2D embedding (linear or non-linear) cleanly separates Class 0 vs Class 1. The class boundary is not a low-dimensional manifold — it requires the full 15–21 dimensional representation and a non-linear classifier.

### The 150 features reduce to ~50 independent signals

Union-Find clustering at r ≥ 0.99 identifies **7 correlation clusters** accounting for 107 of 150 features. The dominant cluster alone contains 75 near-perfectly-correlated features — essentially 75 rescaled copies of the same underlying variable. This is why PC1 captures 66.95% of variance with uniform loadings (entropy = 4.69, near-maximum for the cluster).

### PCA vs raw features for classification (5-fold CV, n=20K)

| Classifier | Best config | AUC |
|---|---|---:|
| Logistic Regression | Raw | 0.7220 |
| Random Forest | Raw | 0.7229 |
| MLP | PCA-31 | 0.7196 |

**PCA does not improve classification accuracy** — it matches raw features at sufficient k. The value of PCA is engineering efficiency: **PCA-15 delivers equivalent AUC with 90% fewer features**, cutting memory, training time, and inference latency.

### Clustering: all methods converge on k=2

Six clustering methods were evaluated with four internal validity indices plus bootstrap stability:

| Method | Silhouette | Notes |
|---|---:|---|
| **KMeans++ (k=2)** | **0.6165** | Best overall |
| Agglomerative Ward (k=2) | 0.6157 | Best if hierarchy needed |
| HDBSCAN | 0.4526 | Finds 3 groups + 2.4% noise |
| Agglomerative Complete/Average | 0.5321 | Poor CH; unequal cluster sizes |
| DBSCAN (best config) | 0.4841 | 13.1% noise; useful for outlier detection |
| OPTICS | −0.5143 | Not viable for this dataset |

KMeans, MiniBatchKMeans, BisectingKMeans, GMM, and Agglomerative Ward all produce **identical scores** (silhouette=0.6165, DB=0.5889, CH=18,624) — strong evidence the k=2 partition is a genuine geometric structure, not an algorithm artifact. **Bootstrap stability (30 iterations, 80% subsample): ARI > 0.85 for k=2–6**, but silhouette collapses for k>2.

### The k=2 split is driven by a single feature

A depth-1 decision tree achieves **AUC = 1.0** on cluster prediction with the rule `claim_to_premium_ratio <= -499.50`. This explains:
- Why KS = 1.000 for all top features (they're all correlated with `claim_to_premium_ratio`)
- Why bootstrap ARI = 1.0000 (the split is deterministic)
- Why RF permutation importance = 0.0 (any single feature can be removed; the others encode the same rule)
- Why the centroid heatmap shows ±1.00 on every feature (perfect anti-correlation imposed by the split)

### The geometric split does NOT align with the fraud target

Both clusters are ~75–77% Class 1 (vs the 76.1% dataset baseline). ARI ≈ 0 and NMI ≈ 0 across all k values. **Cluster labels should not be used as model features** — they carry no classification-relevant information. The split is a useful business segmentation: "standard claims" (Cluster 0, larger) vs "high-premium cross-sell clients" (Cluster 1, smaller).

### Sub-cluster analysis surfaces a hidden high-risk sub-population

Within each macro-cluster, a ~20% sub-population shows substantially elevated class-1 rate:
- Macro-0 sub-group: 92.0% Class 1 (vs 72.3% for the rest of Macro-0)
- Macro-1 sub-group: 89.7% Class 1 (vs 71.1% for the rest of Macro-1)

These sub-groups appear as edge points at the extremities of each macro-cluster's PCA projection — likely claims with extreme `severity_index` or `recovery_probability` values.

📄 **Full report:** [`2. Clustering and Dimensionality Reduction Analysis.pdf`](./2.%20Clustering%20and%20Dimensionality%20Reduction%20Analysis.pdf)

---

## Stage 3 — Reinforcement Learning (Claims Decisioning Agent)

**Task:** Train an agent to choose an _action_ on each claim (not just a prediction), using the supervised model's output as reward-shaping signal.

### Environment design

| Component | Specification |
|---|---|
| **State (20-dim)** | 15 PCA components (from Stage 2) + 5 raw binary EDA flags (`has_prior_denial`, `has_customer_value`, `is_fast_claim`, `coverage_limit_high`, `is_zero_touchpoints`), quantized into 5 equal-frequency bins → sparse Q-table |
| **Actions (4)** | Fast-track Approve · Request Documentation · Manual Review · Deny |
| **Reward** | Correct approve: +10 · Correct deny: +8 · Wrong fast-track (label=0, action=0): **−20 − 5×(1−p)** · Wrong deny (label=1, action=3): −3 · Delay actions: −1 + uncertainty bonus |
| **Training** | Tabular Q-learning, ε-greedy (linear decay 1.0 → 0.05), cosine-annealed LR (0.35 → 0.05), 80,000 episodes |

The `−5×(1−p)` term — where `p` is the supervised model's predicted probability — is the **key reward-shaping mechanism** that solves the flat-reward exploration problem. Without it, the agent has no signal about which wrong fast-tracks are riskier than others.

The Q-table saturates at ~9,025 visited states out of a theoretical 5²⁰ ≈ 9.5×10¹³ — confirming the discretization converged and the agent explored the relevant state space.

### Policy comparison

| Policy | Avg Reward | Accuracy | AUC | F1 (Class 0) | Risky Claims Fast-Tracked |
|---|---:|---:|---:|---:|---:|
| **Q-Learning** | **+2.39** | 0.75 | 0.52 | 0.12 | 557 |
| Always Approve | +2.24 | 0.76 | 0.50 | 0.00 | 705 |
| Supervised RF (as policy) | +1.10 | 0.76 | 0.51 | 0.07 | 24 |
| Random | +0.38 | 0.63 | 0.50 | 0.23 | 191 |
| Always Review | +0.07 | 0.76 | 0.50 | 0.00 | 0 |

### The critical trade-off — and honest assessment

Q-Learning achieves the highest average reward (+2.39), beating even Always Approve (+2.24) despite similar accuracy. But the **risky fast-track count exposes the fundamental RL limitation**: the agent fast-tracks 557 risky claims vs only 24 for the supervised RF.

**Why this happens:** The supervised RF optimizes classification (symmetric misclassification minimization) → excellent fraud detection. The RL agent optimizes cumulative reward → it learns that the reward-maximizing policy under this specific reward function is "approve most, deny the clearly suspicious." Since 76% of claims genuinely deserve approval, a mostly-approve policy earns high cumulative reward while still missing substantial fraud.

This makes RL more flexible (policy adapts if the reward function changes) but also more dangerous if fraud prevention is the primary business objective. The KS=1.0 between FP and FN probability distributions on the supervised model confirms the RL agent does differentiate — claims it sends to review or deny have meaningfully lower P(approve) than those it fast-tracks.

### Recommended deployment: hybrid

Use the supervised model's probability as part of the RL state (as done here via reward shaping), and deploy RL as the **decision layer** optimizing operational efficiency on top of the supervised signal — not as a replacement for it.

📄 **Full report:** [`3. RL.pdf`](./3.%20RL.pdf)

---

## Skill Matrix

What this project demonstrates, organized by competency:

| Competency | Evidence in this project |
|---|---|
| **Exploratory Data Analysis** | Bimodal missingness diagnosis, MNAR proof via missingness-only logistic regression (AUC=0.60), hierarchical clustering of missingness patterns |
| **Statistical testing** | t-tests, ANOVA F-stats, chi-square, Mann-Whitney U, KS, mutual information — applied systematically to every feature |
| **Feature engineering** | 8 binary thresholds derived from CDF crossover analysis, missingness-cluster features (+0.005 AUC), interaction terms (`denial_x_processing`) |
| **Supervised learning** | LightGBM, CatBoost, XGBoost, RandomForest, linear baselines (LogReg/SGD/LDA), H2O AutoML, stacking ensembles |
| **Hyperparameter optimization** | 6 strategies benchmarked: Optuna TPE, Hyperopt, PBT, NSGA-II/CMA-ES (multi-objective Pareto), GA, Hyperband |
| **Model interpretation** | SHAP on errors (FP vs FN), permutation importance, EDA-vs-error cross-reference table |
| **Unsupervised learning** | PCA / Kernel PCA / Factor Analysis / t-SNE / UMAP / NMF / autoencoder (7 DR methods) + KMeans / GMM / Agglomerative (3 linkages) / DBSCAN / HDBSCAN / OPTICS (6 clustering methods) |
| **Cluster validation** | Silhouette, Davies-Bouldin, Calinski-Harabasz, Gap statistic, bootstrap ARI stability (30 iterations) |
| **Reinforcement learning** | Custom tabular Q-learning implementation with sparse hash-map Q-table, ε-greedy with linear decay, cosine-annealed LR, reward shaping from supervised probability |
| **Reproducibility / engineering** | Stratified K-fold CV, OOF prediction tracking, experiment tracking, configuration-driven workflows |
| **Honest reporting** | Cross-referenced EDA predictions against test errors; explicit recommendation tables; RL limitations acknowledged |

---

## Tech Stack

- **Languages / Core:** Python, pandas, NumPy
- **Gradient Boosting:** LightGBM, CatBoost, XGBoost, scikit-learn (HistGradientBoosting), H2O AutoML, AutoGluon, PyCaret
- **Hyperparameter Optimization:** Optuna (TPE, NSGA-II, CMA-ES, Hyperband), Hyperopt
- **Dimensionality Reduction:** scikit-learn (PCA, Kernel PCA, Factor Analysis, NMF), t-SNE, UMAP, HDBSCAN, custom autoencoder (Keras/PyTorch)
- **Clustering:** scikit-learn (KMeans, MiniBatchKMeans, BisectingKMeans, GMM, Agglomerative, DBSCAN, OPTICS), HDBSCAN
- **Explainability:** SHAP
- **Reinforcement Learning:** custom Q-learning implementation (tabular, sparse hash-map Q-table, ε-greedy + cosine-annealed LR)
- **Visualization:** Matplotlib, Seaborn

---

## Repository Structure

```
insurance_claims_decisioning/
├── 1. supervised learning.pdf                          # Stage 1 full report (6.4 MB)
├── 2. Clustering and Dimensionality Reduction Analysis.pdf  # Stage 2 full report (7.6 MB)
├── 3. RL.pdf                                           # Stage 3 full report (283 KB)
├── README.md
├── sample_data.csv                                     # Tiny sample for sanity checks
│
├── eda.ipynb                                           # Exploratory data analysis (49 MB, with figures)
├── dim_reduction.ipynb                                 # PCA / Kernel PCA / t-SNE / UMAP / FA / NMF / AE
├── clustering.ipynb                                    # KMeans / GMM / Agglomerative / DBSCAN / HDBSCAN / OPTICS
├── rl.ipynb                                            # Q-learning agent + policy evaluation
├── time_series.ipynb                                   # Time-series exploration
├── bigData_dask.ipynb                                  # Dask scalability experiment
│
├── src/                                                # Shared utilities
│   ├── config.py
│   ├── data_loader.py
│   ├── data_splitter.py                                # Fold creation for OOF + stacking
│   ├── training_utils.py
│   ├── optuna_utils.py                                 # Multi-objective Pareto search (NSGA-II / CMA-ES)
│   ├── oof_manager.py                                  # Out-of-fold prediction tracking
│   ├── evaluation_utils.py                             # Metrics + calibration + error analysis
│   ├── postprocessing_utils.py
│   ├── experiment_tracker.py
│   ├── visualization_utils.py
│   └── utils.py
│
├── models/                                             # Supervised-learning notebooks + outputs
│   ├── feature_engineering.ipynb                       # Preprocessing + feature engineering + folds
│   ├── base_models.ipynb
│   ├── linear_models.ipynb                             # LogReg / SGD / LDA baselines
│   ├── tree_models.ipynb
│   ├── lgbm_optuna.ipynb                               # LightGBM + Optuna + error analysis
│   ├── catboost.ipynb                                  # CatBoost + error analysis (best model)
│   ├── xgboost_obtuna.ipynb                            # XGBoost + Optuna
│   ├── rf_optuna.ipynb
│   ├── adaboost_optuna.ipynb
│   ├── histgb_optuna.ipynb
│   ├── stacking.ipynb
│   ├── ensamble.ipynb                                  # Final ensembling
│   ├── tabpfn.ipynb
│   ├── tfdf.ipynb
│   ├── autoglean.ipynb                                 # AutoGluon
│   ├── pycaret.ipynb
│   ├── h2o.ipynb                                       # H2O AutoML leaderboard
│   ├── kaggle.ipynb
│   ├── lgbm_cnn.ipynb
│   ├── rapids_cuML.ipynb                               # GPU-accelerated clustering experiment
│   ├── model.ipynb
│   └── utils.py
│
├── outputs/                                            # Trained models + OOF predictions + experiment metadata
├── data/                                               # Dataset (gitignored — see sample_data.csv)
└── .gitignore
```

---

## How to Run

The notebooks are designed to be run in order — each stage consumes the previous stage's outputs.

```bash
git clone https://github.com/ismailSadouki/insurance_claims_decisioning.git
cd insurance_claims_decisioning

# Stage 1 — Supervised Learning
jupyter notebook eda.ipynb                              # EDA + missingness analysis
jupyter notebook models/feature_engineering.ipynb       # preprocessing + folds
jupyter notebook models/lgbm_optuna.ipynb               # or catboost.ipynb / xgboost_obtuna.ipynb / linear_models.ipynb
jupyter notebook models/ensamble.ipynb                  # final ensembling

# Stage 2 — Clustering & Dimensionality Reduction
jupyter notebook dim_reduction.ipynb
jupyter notebook clustering.ipynb

# Stage 3 — Reinforcement Learning
jupyter notebook rl.ipynb
```

> The project is deliberately split across multiple notebooks rather than one monolith — this matches an existing plug-and-play workflow and makes it easier to iterate on one stage without re-running the others.

---

## Author

**Ismail Sadouki** — [github.com/ismailSadouki](https://github.com/ismailSadouki)
