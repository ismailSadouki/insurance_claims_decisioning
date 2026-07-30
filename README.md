# Insurance Claims Decisioning — Supervised Learning, Clustering/Dimensionality Reduction & Reinforcement Learning

An end-to-end machine learning study on a large-scale, heavily-obfuscated insurance claims dataset (~113K rows, 150+ features). The project moves through three connected stages: **predicting** claim outcomes, **understanding** the geometry of the feature space, and **deciding** what action to take on each claim using reinforcement learning shaped by the supervised model's output.

> **Note on the data:** All column names in this dataset are fictitious. This is provable statistically — several "unrelated" columns (e.g. a count and a quality score, a rate and a count) show correlations above r = 0.99, which is only possible if the underlying values are shared/synthetic rather than the named concepts. Every claim and decision in this project is therefore backed by statistical evidence (KS statistic, AUC direction, mutual information, correlation structure, t-tests, ANOVA, chi-square) rather than by what a column name suggests.

---

# **Full Write-ups (PDF reports)**

- [`supervised_learning.pdf`](https://github.com/ismailSadouki/insurance_claims_decisioning/blob/main/1.%20supervised%20learning.pdf): full EDA, feature engineering, and modeling report.
- [`Clustering_and_Dimensionality_Reduction_Analysis.pdf`](https://github.com/ismailSadouki/insurance_claims_decisioning/blob/main/2.%20Clustering%20and%20Dimensionality%20Reduction%20Analysis.pdf): full PCA/clustering report.
- [`RL.pdf`](https://github.com/ismailSadouki/insurance_claims_decisioning/blob/main/3.%20RL.pdf): full RL environment/agent report.


## Project Files (In Order)

**Supervised Learning**

- [`models/feature_engineering.ipynb`](https://github.com/ismailSadouki/x42/blob/main/models/feature_engineering.ipynb) — data preprocessing and feature engineering, plus data splitting and fold creation used later for modeling and stacking.
- [`models/lgbm_optuna.ipynb`](https://github.com/ismailSadouki/x42/blob/main/models/lgbm_optuna.ipynb) — LightGBM with Optuna, several optimization strategies, and error analysis.
- [`models/catboost.ipynb`](https://github.com/ismailSadouki/x42/blob/main/models/catboost.ipynb) — CatBoost + error analysis.
- [`models/xgboost_optuna.ipynb`](https://github.com/ismailSadouki/x42/blob/main/models/xgboost_obtuna.ipynb) / [`models/linear_models.ipynb`](https://github.com/ismailSadouki/x42/blob/main/models/linear_models.ipynb) — XGBoost and several other baseline/comparison models.
- [`models/ensamble.ipynb`](https://github.com/ismailSadouki/x42/blob/main/models/ensamble.ipynb) — final ensembling.

All models are saved with their respective parameters and OOF predictions in [`outputs/experiments`](https://github.com/ismailSadouki/x42/tree/main/outputs/experiments).

**Dimensionality Reduction**

- [`dim_reduction.ipynb`](https://github.com/ismailSadouki/x42/blob/main/dim_reduction.ipynb)

**Clustering**

- [`clustering.ipynb`](https://github.com/ismailSadouki/x42/blob/main/clustering.ipynb)

**Reinforcement Learning**

- [`rl.ipynb`](https://github.com/ismailSadouki/x42/blob/main/rl.ipynb)




---

## 1. Supervised Learning : Fraud/Claim Classification

**Task:** Binary classification on ~90.8K training / ~22.7K test insurance claims (112 numerical + 19 categorical features, 76.1% / 23.9% class split, 3.17:1 imbalance).

**Highlights:**

- **Deep EDA**: missingness is bimodal at the row level (rows are either ~0% or ~100% missing) and is _not_ random. a logistic regression trained purely on the missingness pattern (no real feature values) scores AUC 0.60, and hierarchical clustering of missingness patterns produced engineered features that lifted AUC from 0.7038 → 0.7089.
- **Feature engineering**: zero-inflated features converted to binary indicators, CDF-crossover analysis used to derive optimal binary thresholds (e.g. `has_prior_denial`, `high_coverage_limit`, `is_zero_touchpoints`), and high-cardinality categoricals (18K+ unique claim codes) handled via CatBoost's native ordered target statistics.
- **Modeling**: LightGBM (Optuna-tuned, multi-objective Pareto selection balancing pAUC vs. train/val gap), CatBoost, XGBoost, and an H2O AutoML leaderboard were all benchmarked against linear baselines (Logistic Regression, SGD, LDA), which topped out ~3% AUC lower. confirming the problem is non-linear.
- **Best model: CatBoost** - OOF pAUC 0.630, test ROC-AUC 0.7755, PR-AUC 0.914.
- **Error analysis**: SHAP on false positives/negatives, Mann-Whitney U separation tests, and a full EDA-vs-error cross-reference to validate that the model's mistakes are explainable, plus concrete recommendations (probability recalibration, threshold re-optimization, dropping 3,416 redundant feature pairs with r > 0.97).

|Metric|Value|
|---|---|
|Model|CatBoost|
|ROC-AUC|0.7755|
|PR-AUC|0.9140|
|F1 (Class 0 / Class 1)|0.5253 / 0.7872|
|Brier Score|0.1778|

---

## 2. Clustering & Dimensionality Reduction

**Task:** Understand the intrinsic geometry of the 150-feature space and test whether natural clusters align with the classification target.

**Dimensionality reduction:**

- PCA: just **4 components explain 80%** of total variance; PC1 alone explains 66.95%, revealing massive feature redundancy (nearly half of all 11,175 feature pairs exceed |r| ≥ 0.85, and the 150 raw features reduce to only ~50 truly independent signals).
- Kernel PCA, t-SNE, UMAP, Factor Analysis, NMF, and an autoencoder were all benchmarked — **none meaningfully improve class separability over linear PCA**, indicating the classification boundary is not a simple low-dimensional manifold.
- **Recommendation:** PCA-15 to PCA-21 for downstream modeling (95% variance, 90% fewer features, no accuracy loss vs. raw features); UMAP for visualization.

**Clustering:**

- KMeans, GMM, Agglomerative (Ward/Complete/Average), DBSCAN, HDBSCAN, and OPTICS were all evaluated with Silhouette, Davies-Bouldin, Calinski-Harabasz, Gap Statistic, and bootstrap stability (ARI).
- Every method converges on the same robust **k = 2** geometric split (Silhouette 0.6165 for KMeans++), driven entirely by a single feature threshold (`claim_to_premium_ratio`) — a decision tree of depth 1 separates the two clusters with **AUC = 1.0**.
- Critically, **this geometric split does not align with the fraud/claim target** (ARI ≈ 0, NMI ≈ 0) — both clusters are ~75–77% Class 1 — so **cluster labels should not be used as model features**, but the split is a useful business segmentation ("standard claims" vs. "high-premium cross-sell clients").
- Sub-cluster analysis surfaces a ~20% high-risk sub-population inside _each_ macro-cluster with a substantially elevated positive rate (92.0% / 89.7% vs. ~72–79% baseline).

---

## 3. Reinforcement Learning — Claims Decisioning Agent

**Task:** Train an agent to choose an _action_ on each claim (not just a prediction), using the supervised model's output as reward-shaping signal.

- **State (20-dim):** 15 PCA components (from Part 2) + 5 raw high-signal binary flags identified in the EDA (`has_prior_denial`, `has_customer_value`, `is_fast_claim`, `coverage_limit_high`, `is_zero_touchpoints`), quantized into 5 equal-frequency bins → sparse Q-table (~9,025 states visited out of a theoretical 5²⁰).
- **Actions:** Fast-track Approve · Request Documentation · Manual Review · Deny.
- **Reward shaping:** correct approvals/denials are rewarded; a _wrong_ fast-track approval is penalized proportionally to the supervised model's confidence (`−20 − 5×(1−p)`), which is what solves the flat-reward exploration problem.
- **Training:** tabular Q-learning, ε-greedy (linear decay 1.0 → 0.05) and cosine-annealed learning rate (0.35 → 0.05), 80,000 episodes.

|Policy|Avg. Reward|Accuracy|AUC|F1 (Class 0)|Risky Claims Fast-Tracked|
|---|---|---|---|---|---|
|**Q-Learning**|**+2.39**|0.75|0.52|0.12|557|
|Always Approve|+2.24|0.76|0.50|0.00|705|
|Supervised RF (as policy)|+1.10|0.76|0.51|0.07|24|
|Random|+0.38|0.63|0.50|0.23|191|
|Always Review|+0.07|0.76|0.50|0.00|0|

**Key takeaway:** the RL agent maximizes cumulative reward, not classification accuracy — it learns an "approve most, deny the clearly suspicious" policy that beats naive baselines on reward but is _riskier_ than a pure supervised classifier (557 vs. 24 risky claims fast-tracked). The recommended production setup is a **hybrid**: use the supervised model's probability as part of the RL state/reward (as done here), and deploy RL as a decision layer optimizing operational efficiency on top of the supervised signal — not as a replacement for it.

---

## Tech Stack

- **Languages/Core:** Python, pandas, NumPy
- **Modeling:** LightGBM, CatBoost, XGBoost, scikit-learn, H2O AutoML
- **Tuning:** Optuna (TPE, multi-objective / Pareto-front selection)
- **Dimensionality Reduction / Clustering:** scikit-learn (PCA, Kernel PCA, Factor Analysis, NMF), t-SNE, UMAP, HDBSCAN, an autoencoder (Keras/PyTorch)
- **Explainability:** SHAP
- **Reinforcement Learning:** custom Q-learning implementation (tabular, sparse hash-map Q-table)
- **Visualization:** Matplotlib, Seaborn

---

## How to Run

```bash
git clone https://github.com/ismailSadouki/insurance_claims_decisioning.git
cd x42

# each stage lives in its own notebook for parallel iteration and reusability
jupyter notebook models/feature_engineering.ipynb   # start here: preprocessing + folds
jupyter notebook models/lgbm_optuna.ipynb           # or catboost.ipynb / xgboost_optuna.ipynb / linear_models.ipynb
jupyter notebook models/ensamble.ipynb              # final ensembling

jupyter notebook dim_reduction.ipynb
jupyter notebook clustering.ipynb
jupyter notebook rl.ipynb
```

> The project is deliberately split across multiple notebooks rather than one monolith — this matches an existing plug-and-play workflow and makes it easier to iterate on one stage without re-running the others.

---

## Author

**Ismail Sadouki** — [github.com/ismailSadouki](https://github.com/ismailSadouki)