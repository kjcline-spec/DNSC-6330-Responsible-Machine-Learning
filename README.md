# DNSC 6330 Responsible Machine Learning

This repository contains coursework, assignments, and projects for DNSC 6330.

## Structure
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

## Data Source

ProPublica COMPAS Analysis Dataset:  
https://github.com/propublica/compas-analysis

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
#### AI Use Statement
I used AI as a learning aid on this assignment. Specifically, 
I used it to talk through my understanding of the R-to-Python translation, 
work through debugging errors as they arose, and verify that my Python 
outputs were conceptually equivalent to the R workflow. All code was reviewed 
and validated by me for accuracy and alignment with the lecture material.

# HW2: COMPAS Replacement Model — Explainability and Governance Analysis

## Purpose

This notebook evaluates a COMPAS replacement model using modern explainability tools, translating Lecture 02 concepts into a full analytical workflow. The goal is not only to explain model predictions, but to assess whether those explanations are sufficient for responsible deployment in a high-stakes setting.

The analysis applies SHAP, LIME, and counterfactual methods (DiCE) to understand model behavior, identify potential proxy effects, and evaluate the consistency and actionability of explanations.

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
- `xgboost` or `sklearn` — model training (gradient-boosted tree)  
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

- SHAP identifies `priors_count` and `age` as dominant drivers of model predictions, consistent across the test set.

- LIME and SHAP generally agree on high-risk individuals, but diverge for lower-risk cases, particularly on categorical features.

- This divergence highlights a key governance issue:
  explanations are method-dependent, which can undermine consistency and due process.

- Counterfactual analysis reveals different recourse pathways across individuals and groups, suggesting the model applies different decision logic depending on feature combinations.

- No counterfactuals required changes to immutable features, but differences in required changes across groups suggest potential proxy effects.

---

## Method Limitations

- SHAP provides additive feature attribution but is not causal. A low SHAP value for a feature does not imply the model is independent of that feature in a broader sense.

- LIME optimizes local fidelity, not global fidelity. Its explanations depend heavily on the chosen neighborhood and may not generalize.

- Counterfactual methods produce mathematically valid solutions, but not all are realistic or actionable in practice.

- Explanations alone do not guarantee fairness or trustworthiness — they are diagnostic tools, not validation.

---

## Governance Implications

- Model evaluation must extend beyond predictive performance to include fairness, consistency, and recourse.

- Differences between explanation methods introduce risk in how decisions are justified to affected individuals.

- Counterfactual outputs should be audited to ensure recommendations are realistic and ethically appropriate.

- Transparency tools must be embedded within a broader governance framework, including monitoring, audit processes, and documented review procedures.

---

## Reproducibility Notes

- The workflow is fully reproducible and does not require local files  
- Minor variation in results may occur due to:
  - stochastic elements in model training
  - differences in library implementations  
- These do not affect the overall conclusions  

---

## Responsible ML Context

This analysis demonstrates that explainability is necessary but not sufficient for responsible machine learning. Even when models are transparent, they may still encode proxy relationships and produce inconsistent or non-actionable explanations.

As discussed in Lecture 02, transparency tools should be used to diagnose model behavior, not to justify deployment decisions in isolation.

#### AI Use Statement
I used AI as a learning aid on this assignment. Specifically, 
I used it to talk through my understanding of the R-to-Python translation, 
work through debugging errors as they arose, and verify that my Python 
outputs were conceptually equivalent to the R workflow. All code was reviewed 
and validated by me for accuracy and alignment with the lecture material.

