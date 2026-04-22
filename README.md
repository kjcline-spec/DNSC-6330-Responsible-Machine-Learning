# DNSC 6330 Responsible Machine Learning

This repository contains coursework, assignments, and projects for DNSC 6330.

---

## Table of Contents
- [HW1: COMPAS Analysis](#hw1-compas-analysis)
  - [Purpose](#purpose)
  - [Workflow Overview](#workflow-overview)
  - [Python Libraries Used](#python-libraries-used)
  - [Instructions for Reproducing Results](#instructions-for-reproducing-results)
  - [Key Findings](#key-findings)
  - [Reproducibility Notes](#reproducibility-notes)
  - [Responsible ML Context](#responsible-ml-context)
  - [AI Use Statement](#ai-use-statement)
- [HW2: Explaining the COMPAS Replacement Model](#hw2-explaining-the-compas-replacement-model)
  - [Purpose](#purpose-1)
  - [Workflow Overview](#workflow-overview-1)
  - [Python Libraries Used](#python-libraries-used-1)
  - [Instructions for Reproducing Results](#instructions-for-reproducing-results-1)
  - [Key Findings](#key-findings-1)
  - [Method Limitations](#method-limitations)
  - [Governance Implications](#governance-implications)
  - [Reproducibility Notes](#reproducibility-notes-1)
  - [Responsible ML Context](#responsible-ml-context-1)
  - [AI Use Statement](#ai-use-statement-1)
- [HW3: Algorithmic Bias Measurement — COMPAS](#hw3-algorithmic-bias-measurement--compas)
  - [Purpose](#purpose-2)
  - [Workflow Overview](#workflow-overview-2)
  - [Python Libraries Used](#python-libraries-used-2)
  - [Instructions for Reproducing Results](#instructions-for-reproducing-results-2)
  - [Key Findings](#key-findings-2)
  - [Limitations](#limitations)
  - [Responsible ML Context](#responsible-ml-context-2)
  - [AI Use Statement](#ai-use-statement-2)
- [HW4: Robustness, Generalization, and Dataset Drift](#hw4-robustness-generalization-and-dataset-drift)
  - [Purpose](#purpose-3)
  - [Workflow Overview](#workflow-overview-3)
  - [Python Libraries Used](#python-libraries-used-3)
  - [Instructions for Reproducing Results](#instructions-for-reproducing-results-3)
  - [Key Findings](#key-findings-3)
  - [Responsible ML Context](#responsible-ml-context-3)
  - [AI Use Statement](#ai-use-statement-3)
- [HW5: Adversarial ML — Evasion, Poisoning, and Membership Inference](#hw5-adversarial-ml--evasion-poisoning-and-membership-inference)
  - [Purpose](#purpose-4)
  - [Workflow Overview](#workflow-overview-4)
  - [Python Libraries Used](#python-libraries-used-4)
  - [Instructions for Reproducing Results](#instructions-for-reproducing-results-4)
  - [Key Findings](#key-findings-4)
  - [Responsible ML Context](#responsible-ml-context-4)
  - [AI Use Statement](#ai-use-statement-4)

---

## Structure
### HW1: COMPAS ANALYSIS
#### Purpose, Workflow Overview, Python Libraries, Instructions for Reproducing Results, Key Findings, Reproducibility Notes, Responsible ML Context & AI Use Statement

### HW2: EXPLAINING THE COMPAS REPLACEMENT MODEL
#### Purpose, Workflow Overview, Python Libraries, Instructions for Reproducing Results, Key Findings, Method Limitations, Governance Implications, Reproducibility Notes, Responsible ML Context & AI Use Statement

### HW3: ALGORITHMIC BIAS MEASUREMENT — COMPAS
#### Purpose, Workflow Overview, Python Libraries, Instructions for Reproducing Results, Key Findings, Limitations, Responsible ML Context & AI Use Statement

---

# HW1: COMPAS Analysis
## COMPAS Recidivism Risk Score Analysis — Python Replication

## Purpose
This notebook replicates ProPublica's analysis of the COMPAS recidivism
risk scoring algorithm, translating the original R workflow from DNSC 6330
Lecture 01 into Python. The analysis performs exploratory data analysis on
the Broward County defendant dataset, builds a logistic regression model to
test for racial bias in COMPAS scores, and evaluates model diagnostics
including confusion matrices and FPR/FNR disparity by race.

---

## Workflow Overview

This analysis follows the same structure as the original R workflow from Lecture 01:

1. **Data Loading and Cleaning**
   - Load COMPAS dataset from ProPublica
   - Apply filtering rules (date range, valid scores, charge types)
   - Create derived variables and factor encodings

2. **Exploratory Data Analysis (EDA)**
   - Summary statistics of key variables (age, priors_count, race)
   - Distribution of COMPAS risk scores
   - Visualization of recidivism outcomes

3. **Model Development**
   - Logistic regression model using `statsmodels`
   - Predict COMPAS high-risk classification
   - Replicates R `glm(..., family=binomial)` specification

4. **Model Evaluation and Diagnostics**
   - Confusion matrix (overall and by race)
   - Performance metrics: accuracy, precision, recall
   - Fairness metrics: FPR and FNR by race
   - Disparity analysis (Black vs White defendants)

The goal is to ensure conceptual equivalence with the R implementation, not exact numerical matching.

---

## Python Libraries Used

- `pandas` — data loading, filtering, and manipulation
- `numpy` — numerical operations
- `matplotlib` — visualizations
- `seaborn` — plot styling
- `statsmodels` — logistic regression (equivalent to R glm with family=binomial)
- `scipy` — statistical calculations
- `math` — supporting calculations

---

## Instructions for Reproducing Results

1. Open the notebook in Google Colab or any Jupyter environment
2. Run all cells from top to bottom in order
3. No local data files are needed — the dataset loads directly from the ProPublica GitHub repository via URL
4. All cells should run without errors and reproduce the outputs shown

---

## Key Findings

- The model demonstrates racial disparity in error rates:
  - Black defendants have a higher false positive rate (FPR)
  - White defendants have a higher false negative rate (FNR)

- The model is approximately calibrated across groups, but this comes at the cost of unequal error rates.

- This reflects a core tradeoff discussed in Lecture 01:
  calibration and equal error rates cannot both be satisfied simultaneously.

- The results highlight the alignment problem:
  optimizing prediction accuracy does not guarantee equitable outcomes across groups.

---

## Reproducibility Notes

- The workflow is fully self-contained and does not require local files
- Minor numerical differences from the R implementation may occur due to:
  - differences in library implementations (R vs Python)
  - floating point precision
- These differences do not affect the overall conclusions or interpretation

---

## Responsible ML Context

This analysis demonstrates that machine learning systems must be evaluated as part of a broader system, not just based on accuracy. Differences in error rates across groups have real-world implications in high-stakes settings like criminal justice.

As discussed in Lecture 01, removing protected attributes alone does not eliminate bias, as proxy variables can encode similar information.

---

## AI Use Statement

I used AI as a learning aid on this assignment. Specifically,
I used it to talk through my understanding of the R-to-Python translation,
work through debugging errors as they arose, and verify that my Python
outputs were conceptually equivalent to the R workflow. All code was reviewed
and validated by me for accuracy and alignment with the lecture material.

---

# HW2: Explaining the COMPAS Replacement Model

## Purpose

This notebook evaluates a COMPAS replacement model using modern explainability
tools, translating Lecture 02 concepts into a full analytical workflow. The goal
is not only to explain model predictions, but to assess whether those explanations
are sufficient for responsible deployment in a high-stakes setting.

The analysis applies SHAP, LIME, and counterfactual methods (DiCE) to understand
model behavior, identify potential proxy effects, and evaluate the consistency and
actionability of explanations.

---

## Workflow Overview

This analysis follows the structure introduced in Lecture 02:

1. **Model Explanation (SHAP)**
   - Compute SHAP values on the test set
   - Generate beeswarm summary plot
   - Generate waterfall plots for:
     - Highest-risk and lowest-risk individuals
     - Across different racial groups

2. **Local Explanation Comparison (LIME vs SHAP)**
   - Apply LIME to the same selected individuals
   - Compare feature attributions between methods
   - Identify areas of agreement and divergence
   - Interpret implications for explanation consistency

3. **Counterfactual Analysis (DiCE)**
   - Generate counterfactuals for each individual
   - Identify minimal feature changes required to flip predictions
   - Flag any counterfactuals involving immutable attributes
   - Evaluate plausibility and actionability of recommendations

4. **Governance Assessment**
   - Interpret model behavior through explanations
   - Identify risks related to fairness, consistency, and recourse
   - Provide monitoring and audit recommendations

---

## Python Libraries Used

- `pandas` — data manipulation
- `numpy` — numerical operations
- `matplotlib` — plotting
- `seaborn` — visualization styling
- `scikit-learn` — model training (logistic regression and gradient-boosted tree)
- `shap` — global and local feature attribution
- `lime` — local surrogate explanations
- `dice-ml` — counterfactual generation

---

## Instructions for Reproducing Results

1. Open the notebook in Google Colab or a Jupyter environment
2. Install required packages if needed (e.g., `pip install shap lime dice-ml`)
3. Run all cells from top to bottom
4. Ensure plots render correctly (SHAP plots may require JS support in some environments)

---

## Key Findings

- SHAP identifies `priors_count` and `age` as dominant drivers of model predictions,
  consistent across the test set.

- LIME and SHAP generally agree on high-risk individuals, but diverge for lower-risk
  cases, particularly on categorical features.

- This divergence highlights a key governance issue: explanations are method-dependent,
  which can undermine consistency and due process.

- Counterfactual analysis reveals different recourse pathways across individuals and
  groups, suggesting the model applies different decision logic depending on feature
  combinations.

- No counterfactuals required changes to immutable features, but differences in required
  changes across groups suggest potential proxy effects.

---

## Method Limitations

- SHAP provides additive feature attribution but is not causal. A low SHAP value for a
  feature does not imply the model is independent of that feature in a broader sense.

- LIME optimizes local fidelity, not global fidelity. Its explanations depend heavily on
  the chosen neighborhood and may not generalize.

- Counterfactual methods produce mathematically valid solutions, but not all are realistic
  or actionable in practice.

- Explanations alone do not guarantee fairness or trustworthiness — they are diagnostic
  tools, not validation.

---

## Governance Implications

- Model evaluation must extend beyond predictive performance to include fairness,
  consistency, and recourse.

- Differences between explanation methods introduce risk in how decisions are justified
  to affected individuals.

- Counterfactual outputs should be audited to ensure recommendations are realistic and
  ethically appropriate.

- Transparency tools must be embedded within a broader governance framework, including
  monitoring, audit processes, and documented review procedures.

---

## Reproducibility Notes

- The workflow is fully reproducible and does not require local files
- Minor variation in results may occur due to:
  - stochastic elements in model training
  - differences in library implementations
- These do not affect the overall conclusions

---

## Responsible ML Context

This analysis demonstrates that explainability is necessary but not sufficient for
responsible machine learning. Even when models are transparent, they may still encode
proxy relationships and produce inconsistent or non-actionable explanations.

As discussed in Lecture 02, transparency tools should be used to diagnose model
behavior, not to justify deployment decisions in isolation.

---

## AI Use Statement

I used AI as a learning aid on this assignment. Specifically,
I used it to talk through my understanding of SHAP, LIME, and DiCE concepts
from Lecture 02, work through debugging errors as they arose, and verify that
my Python outputs were conceptually aligned with the lecture material. All code
and written analysis was reviewed and validated by me for accuracy and alignment
with course content.

---

# HW3: Algorithmic Bias Measurement — COMPAS

## Purpose

This notebook applies formal bias measurement methods to the COMPAS recidivism
risk scoring algorithm, translating Lecture 03 concepts into a full analytical
workflow. The goal is to quantify disparate impact across racial and sex groups
using industry-standard metrics, assess statistical and practical significance,
and produce a compliance-ready audit summary.

---

## Workflow Overview

This analysis follows the structure introduced in Lecture 03:

1. **Disparity Metrics — Manual Implementation**
   - Adverse Impact Ratio (AIR) by race and sex
   - Marginal Effect (ME) by race and sex
   - Standardized Mean Difference (SMD) on continuous decile score
   - FPR and FNR by race with two-proportion z-tests

2. **Disparity Metrics — solas-ai Library**
   - Replicate AIR and SMD using `solas_disparity`
   - Confirm results match manual calculations
   - Apply EEOC 80% Rule threshold

3. **Intersectional Analysis**
   - Race × sex subgroup analysis
   - Worst-group AIR identification and interpretation
   - Small cell size flagging per Lecture 03 guidance

4. **Visualization**
   - Publication-quality grouped bar chart of FPR and FNR by race
   - Caucasian reference lines for visual comparison

5. **Compliance Memo**
   - 300-word regulatory summary addressed to a hypothetical federal regulator
   - Findings, metrics, limitations, and monitoring recommendations

---

## Python Libraries Used

- `pandas` — data manipulation
- `numpy` — numerical operations
- `matplotlib` — publication-quality visualizations
- `scipy` — statistical calculations
- `statsmodels` — two-proportion z-tests
- `solas-ai` — industry-standard disparity testing library

---

## Instructions for Reproducing Results

1. Open the notebook in Google Colab or any Jupyter environment
2. Run all cells from top to bottom in order
3. No local data files are needed — the COMPAS dataset loads directly from
   the ProPublica GitHub repository via URL
4. Install solas-ai if needed: `pip install solas-ai "kaleido<1.0.0" "plotly<6.0.0"`

---

## Key Findings

- African-American defendants are flagged as high risk at 1.74x the rate of
  Caucasian defendants (AIR = 1.741, SMD = 0.608 — large magnitude).

- FPR for African-American defendants is 0.423 versus 0.220 for Caucasian
  defendants (z = 11.384, p < 0.001) — statistically and practically significant.

- FNR disparity runs in the opposite direction — white defendants miss actual
  recidivists at nearly 2x the rate of Black defendants, illustrating the
  Impossibility Theorem directly.

- Intersectional analysis revealed Hispanic female defendants are flagged at
  only 27% of the Caucasian male reference rate (AIR = 0.270, n = 82) — a
  finding invisible in race-only or sex-only analysis.

- All manual disparity calculations were confirmed against solas-ai library
  output, validating both implementations.

---

## Limitations

- The audit measures association, not causation. Disparities may reflect
  historical policing patterns in the training data rather than model design.

- The Impossibility Theorem means FPR and FNR disparity cannot both be
  eliminated simultaneously when base rates differ across groups.

- Small subgroup sizes for Native American (n = 11) and Asian (n = 31)
  limit the reliability of those estimates.

- This audit addresses statistical disparate impact only. A full compliance
  review would additionally require business necessity analysis and evaluation
  of less discriminatory alternatives.

---

## Responsible ML Context

This analysis demonstrates that bias measurement is a necessary precondition
for responsible deployment. Aggregate accuracy metrics conceal subgroup harms,
and no single metric is sufficient — AIR, FPR, FNR, SMD, and intersectional
analysis each reveal different dimensions of disparity.

As discussed in Lecture 03, transparency without governance is explanation
washing. Disparity metrics must feed into documented audit processes, remediation
workflows, and ongoing monitoring — not treated as one-time compliance checkboxes.

---

## AI Use Statement

I used AI as a learning aid on this assignment. Specifically,
I used it to talk through my understanding of the Lecture 03 bias measurement
concepts, work through the solas-ai library API, debug errors as they arose,
and verify that my outputs were conceptually aligned with the course material.
All code and written analysis was reviewed and validated by me for accuracy
and alignment with course content.

---

# HW4: Robustness, Generalization, and Dataset Drift

## Purpose

This notebook extends the COMPAS analysis from Lectures 01-03 by evaluating
whether the trained models remain reliable beyond the training setting. The goal
is to move from static performance reporting to deployment-defensible evaluation
under distribution drift, generalization failure, spurious correlations, and
robustness stress.

---

## Workflow Overview

This analysis follows the structure introduced in Lecture 04:

1. **Part A: Distribution Drift**
   - PSI and KS tests on numeric features
   - MMD on encoded feature space
   - Score distribution comparison (train vs test)

2. **Part B: Generalization**
   - Train vs test AUC, accuracy, log loss, and Brier score
   - Permutation importance to flag potential overfit reliance

3. **Part C: Spurious-Correlation Probe**
   - Counterfactual swaps on race, sex, and charge degree
   - Mean absolute probability shift as sensitivity measure

4. **Part D: Robustness**
   - Stress test on priors_count (deltas 0, 2, 5, 10)
   - ICE curves for LR and GBT
   - Global sensitivity index

5. **Part E: Slice-Based Evaluation**
   - AUC, FPR, FNR by race, sex, and age subgroups

---

## Python Libraries Used

- `pandas` — data manipulation
- `numpy` — numerical operations
- `matplotlib` — visualizations
- `scikit-learn` — model training, permutation importance, metrics
- `scipy` — KS tests

---

## Instructions for Reproducing Results

1. Open the notebook in Google Colab or any Jupyter environment
2. Run all cells from top to bottom in order
3. No local data files are needed
4. HW4 cells begin after the Lecture 04 setup section and reuse all helper
   functions defined there (psi_numeric, mmd_rbf, evaluate_classifier, etc.)

---

## Key Findings

- PSI and KS tests show stable input distributions between train and test,
  confirming no meaningful covariate shift in the 80/20 split.

- The GBT model shows a larger generalization gap than LR (0.080 vs 0.008),
  indicating greater overfitting risk in the higher-capacity model.

- Counterfactual race swaps produce a measurable shift in predicted
  probabilities, providing evidence of proxy reliance even when race is
  not an explicit model input.

- Stress testing confirms priors_count has high sensitivity: incrementing
  by +5 shifts the predicted high-risk rate substantially for both models,
  with GBT showing greater sensitivity.

- Slice-based evaluation replicates the HW3 FPR disparity finding and
  additionally reveals variation in AUC across age subgroups.

---

## Responsible ML Context

This analysis demonstrates that good average performance is not sufficient
evidence of deployment readiness. Reliable deployment requires that a model
generalize beyond training, remain stable under realistic drift and stress,
and not impose concentrated harm on specific subgroups. As discussed in
Lecture 04, each metric in the audit pipeline answers a governance question
that cannot be skipped.

---

## AI Use Statement

I used AI as a learning aid on this assignment. Specifically, I used it to
talk through my understanding of the Lecture 04 robustness and drift concepts,
work through the helper function implementations, and verify that my outputs
were aligned with the lecture pipeline. All code and written analysis was
reviewed and validated by me for accuracy and alignment with course content.

---

# HW5: Adversarial ML — Evasion, Poisoning, and Membership Inference

## Purpose

This notebook applies adversarial machine learning techniques to the COMPAS
pipeline, translating the Lecture 05 live coding session into a full individual
homework workflow. The goal is to evaluate model vulnerability to deployment-time
evasion, training-time poisoning, and privacy attacks, and to connect attack
findings to governance decisions.

---

## Workflow Overview

This analysis follows the structure introduced in Lecture 05:

1. **Part 1: PGD Evasion Audit**
   - PGD attack across epsilon {0.25, 0.5, 1.0, 2.0} on both LR and GBT
   - FPR by race and AIR at each epsilon
   - Identification of epsilon at which AIR crosses 0.80
   - Comparative vulnerability assessment of LR vs GBT

2. **Part 2: Poisoning Loop with Fairness Monitoring**
   - Label-flip poisoning targeting Caucasian defendants (extending Lecture 05)
   - AUC and AIR degradation curves for both target-race variants
   - Stealth zone identification: AUC drop <= 2pp while AIR outside [0.80, 1.25]
   - PSI-based drift monitor evaluation

3. **Part 3: Membership Inference Depth**
   - Shadow model MI AUC for both LR and GBT
   - Confidence-gap histograms side by side
   - Generalization gap vs MI AUC analysis
   - L2 regularization sweep (C in {0.01, 0.1, 1.0, 10.0}) with MI AUC vs C plot

---

## Python Libraries Used

- `pandas` — data manipulation
- `numpy` — numerical operations
- `matplotlib` — visualizations
- `scikit-learn` — model training, MI shadow models, metrics
- `scipy` — statistical tests

---

## Instructions for Reproducing Results

1. Open the notebook in Google Colab or any Jupyter environment
2. Run all cells from top to bottom in order
3. HW5 cells begin after the Lecture 05 setup section
4. The Lecture 05 setup (Cell 1 and Cell 2) must be run first as HW5 reuses
   the lr, gbt, Xs_tr, Xs_te, y_tr, y_te, r_te, THR, and baseline variables

---

## Key Findings

- PGD evasion shows increasing FPR for African-American defendants as epsilon
  grows, with the fairness gap persisting across all tested values. The GBT
  model was attacked via transfer (LR coefficients as proxy gradient), confirming
  cross-model transferability of adversarial inputs.

- Label-flip poisoning targeting Caucasian defendants produces a different AIR
  trajectory than African-American targeting. In both cases, AUC monitoring alone
  cannot detect the attack because AUC barely moves while fairness metrics degrade.

- PSI-based drift monitors cannot detect label-flip poisoning because feature
  distributions are unchanged. Only subgroup error rate monitoring would catch it.

- LR achieved MI AUC of 0.500 (no detectable leakage) due to its low
  generalization gap. The GBT showed higher leakage risk consistent with its
  larger gap, confirming the generalization-privacy link from Lecture 05.

- The L2 regularization sweep confirms that stronger regularization (lower C)
  reduces MI AUC toward 0.50 at the cost of some predictive accuracy, illustrating
  the privacy-performance tradeoff.

---

## Responsible ML Context

This analysis demonstrates that adversarial robustness and fairness are not
separate concerns. PGD evasion has disparate impact by race, poisoning attacks
can silently corrupt fairness metrics while evading AUC monitoring, and privacy
leakage is directly linked to overfitting. As discussed in Lecture 05, a model
fragile to distribution shift shares the same root cause as a model vulnerable
to adversarial attack: over-reliance on brittle features. Security, robustness,
and fairness require the same structural solution.

---

## AI Use Statement

I used AI as a learning aid on this assignment. Specifically, I used it to
talk through my understanding of the Lecture 05 adversarial attack concepts,
work through the PGD, poisoning, and membership inference implementations,
and verify that my outputs were aligned with the lecture pipeline. All code
and written analysis was reviewed and validated by me for accuracy and
alignment with course content.
