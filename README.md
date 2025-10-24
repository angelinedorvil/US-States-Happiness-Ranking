# US-States-Happiness-Ranking
Predicting regional happiness tiers across U.S. states, a machine learning approach

**Author:** Angeline Dorvil  
**Assignment:** Student Proud Project 
**Date:** Fall 2025  



## Technical Overview

This repository implements an **end-to-end machine learning pipeline** that reproduces and extends the methods of *Janani et al. (2023)* for analyzing quality of life using health and socioeconomic indicators. The implementation focuses on **U.S. state-level well-being prediction** using both **classification and regression** tasks. 



## Repository Structure

US-States-Happiness-Ranking/
│
├── data/                               # Raw and intermediate datasets
│
├── results/                            # All outputs and analysis artifacts
│   ├── norm_predictors/                # Normalized predictor variables (X)
│   ├── norm_targets/                   # Normalized target variables (y)
│   ├── models/                         # Trained model artifacts (.joblib, .json, .csv)
│   └── plots/                          # Saved figures, tables, and visualization outputs
│
├── src/                                # Core source code
│   ├── models/                         # Individual model training scripts
│   │   ├── model_random_forest_regress.py
│   │   ├── model_decision_tree_regress.py
│   │   ├── model_knn_regress.py
│   │   ├── model_svm_regress.py
│   │   ├── model_xgboost_regress.py
│   │   ├── model_ridge_regress.py
│   │   ├── model_lasso_regress.py
│   │   ├── model_linear_regress.py
│   │   ├── model_polyn_regress.py
│   │   └── model_mlp_regress.py
│   │
│   ├── config.py                       # Global constants (paths, random state, etc.)
│   ├── train_all_models.py             # Main orchestration script for training models
│   ├── process_predictor_data.py       # Cleans & normalizes predictor data
│   ├── process_target_data.py          # Cleans & normalizes target (label) data
│   │
│   └── utils/                          # Shared utility modules
│       ├── metrics_utils.py            # Metrics calculation and evaluation helpers
│       ├── evaluation_utils.py         # Cross-validation, comparison, and t-test helpers
│       └── plots_utils.py              # Common plotting functions
│
├── notebooks/                          # Jupyter notebooks for analysis workflow
│   ├── 01_data_exploration.ipynb       # Initial data exploration and cleaning
│   ├── 02_target_processing.ipynb      # Target (y) normalization and prep
│   ├── 03_predictor_processing.ipynb   # Predictor (X) normalization and prep
│   ├── 04_model_training_classification.ipynb
│   ├── 05_model_training_regression.ipynb
│   ├── 06_model_comparison_ttest.ipynb
│   └── 07_feature_importance.ipynb
│
├── requirements.txt                    # Python dependencies
└── README.md                           # Project documentation (this file)



##  How to Run the Code

### 1. Install Environment
```bash
git clone https://github.com/angelinedorvil/US-States-Happiness-Ranking.git
cd US-States-Happiness-Ranking
pip install -r requirements.txt

python train_all_models.py *OR* use Notebooks (see below)
### 2. Datasets

Review notebooks above for detailed information with code examples of each stage. Each notebook can be run. Notebooks can be run **Sequencially** to reach paper outputs.

1. (notebooks/01_data_exploration.ipynb) -> Load raw CHR and FBI data
2. (notebooks/02_target_processing.ipynb) -> Create final_target_index_all_years.csv
3. (notebooks/03_predictor_processing.ipynb) -> Create final_predictor_index_all_years.csv

These files will appear under:
    results/norm_targets/
    results/norm_predictors/

4. (notebooks/04_model_training_classification.ipynb) -> Not implemented for this milestone
5. (notebooks/05_model_training_regression.ipynb) -> Runs train_all_models

Train_all_models file will:
    - Load processed data
    - Train 9 regressors models
    - Save metrics to the results/ directory

6. (notebooks/06_model_comparison_ttest.ipynb) -> Performs and saves analysis of models in plots and results directories
7. (notebooks/07_feature_importance.ipynb) -> Calculate feature importance and permutation importance for models

## Metrics

Some of the regression metrics completed were:
    - R² score
    - Adjusted R²
    - MAE, MSE, RMSE
    - Pearson Correlation Coefficient (r)
    - Matthews Correlation Coefficient (binned approximation)

All regression evaluation is handled by evaluate_regression() in utils/evaluation_utils.py

## Implementation details

1. Normalization:
    - Each metric scaled from 0–1 using min–max normalization.
    - For metrics where “lower is better,” a reverse=True flag flips the scale.

2. Composite Indexing:
    - Target index = weighted sum of normalized metrics + hate crime rate.

3. Data Imputation:
    - Missing predictor values filled using SimpleImputer(strategy="mean").

4. Cross-Validation:
    - KFold (regression) with 5 folds.

5. Reproducibility:
    - All models share a global seed constant from config.RANDOM_STATE.

## References
Janani, S. et al. (2023). Machine Learning for the Analysis of Quality of Life using World Happiness Index and Human Development Indicator.

County Health Rankings & Roadmaps (2015-2025). University of Wisconsin Population Health Institute.

Federal Bureau of Investigation (FBI) Hate Crime Data (1991-2025).