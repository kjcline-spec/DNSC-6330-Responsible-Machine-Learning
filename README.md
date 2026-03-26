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
- HW2: Governance Memo
- Final Project: TBD
