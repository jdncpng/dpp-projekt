# Lung Cancer Survival Prediction
> Predicting patient survival outcomes from a kaggle dataset using machine learning techniques

## Project Overview

**Problem:** using clinical and risk factor data for lung cancer patients, can we predict whether a patient will survive?

**Goal:** Train a binary classifier(`Survived`) that prioritises recall (catching true positives).

**Methods:** Feature Engineering, Random Forest, XGBoost, Logistic Regression via GridSearchCV. Selected Random Forest with F1=0.75, Recall = 0.8919.

**Dataset:** [🫁 Lung Cancer Clinical Dataset (2015–2025)](https://www.kaggle.com/datasets/zkskhurram/lung-cancer-clinical-dataset-20152025) with 1,500 values, and 41 features.

## Setup
clone the repository and install uv

**Running the notebooks**

1. notebooks/01_exploration.ipynb - EDA, class distribution, feature correlations
2. notebooks/02_preparation.ipynb - feature engineering, train/test split -> data/processed
3. notebooks/03_baseline.ipynb - Random Forest (without tuning)
4. notebooks/04_modelling.ipynb - Hyperparameter tuning, model comparison, final model as pickle saved

# Key Findings
| Model               | F1        | ROC-AUC   | Precision | Recall |
| ------------------- | --------- | --------- | --------- | ------ |
| Random Forest       | **0.750** | 0.850     | 0.647     | 0.892  |
| XGBoost             | 0.738     | **0.854** | 0.660     | 0.838  |
| Logistic Regression | 0.742     | 0.844     | 0.635     | 0.892  |


**Top 5 Predictors:** `Cancer_Stage`, `Metastasis`, `Treatment_Surgery`, `Symptom_Count`, `BMI`

# Project Structure

```text
  ├── data/
  │   ├── raw/          # Raw data (not committed)
  │   └── processed/    # Splits + saved model
  ├── notebooks/
  │   ├── pipelines.py  # Shared column, definitions and preprocessing functions      
  │   ├── 01_exploration.ipynb
  │   ├── 02_preparation.ipynb
  │   ├── 03_baseline.ipynb
  │   └── 04_modelling.ipynb
  └── src/core/
      └── data.py       # Kaggle data      
  loading utility
```

# Future recommendations
- test on larger datasets.
- combine with image learning.
- switch to regression tasks
