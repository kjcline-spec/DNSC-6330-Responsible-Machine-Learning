# HW2: COMPAS Replacement Model: Explainability and Governance Analysis
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
