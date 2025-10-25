# US States Happiness Ranking
Predicting regional happiness tiers across U.S. states, a machine learning approach

**Author:** Angeline Dorvil  

**Assignment:** Student Proud Project 

**Date:** Fall 2025  



## Technical Overview

This repository implements an **end-to-end machine learning pipeline** that reproduces and extends the methods of *Janani et al. (2023)* for analyzing quality of life using health and socioeconomic indicators. The implementation focuses on **U.S. state-level well-being prediction** using both **classification and regression** tasks. 



## Repository Structure

<details> <summary><b>data/</b></summary>

 `data/`   ->  Raw source files and intermediate cleaned datasets. 
</details>

<details> <summary><b>results/</b> – Model outputs and analysis artifacts</summary>

 `results/norm_predictors/` -> Normalized predictor variables (X)      

 `results/norm_targets/`    -> Normalized target variables (y)     

 `results/models/`          -> Trained model artifacts (`.joblib`, `.json`, `.csv`) 

 `results/plots/`           -> Saved figures, tables, and visualizations            
</details>

<details> <summary><b>src/</b> – Core source code</summary>

src/config.py	                -> Global constants (paths, random seed, etc.)

src/train_all_models.py	        -> Orchestration script to train and evaluate all models

src/process_predictor_data.py	-> Cleans and normalizes predictor data

src/process_target_data.py	    -> Cleans and normalizes target variable

src/models/	                    -> Individual regression model scripts

├── model_random_forest_regress.py	-> Random Forest Regressor

├── model_decision_tree_regress.py	-> Decision Tree Regressor

├── model_knn_regress.py	        -> k-Nearest Neighbors Regressor

├── model_svm_regress.py	        -> Support Vector Machine Regressor

├── model_xgboost_regress.py	    -> XGBoost Regressor

├── model_ridge_regress.py	        -> Ridge Regressor

├── model_lasso_regress.py	        -> Lasso Regressor

├── model_linear_regress.py	        -> Linear Regressor

├── model_polyn_regress.py	        -> Polynomial Regressor

└── model_mlp_regress.py	        -> Multi-Layer Perceptron (Neural Network) Regressor

src/utils/	                    -> Shared utility modules

├── metrics_utils.py	            -> Metrics and performance calculations

├── evaluation_utils.py	            -> Cross-validation and t-test utilities

└── plots_utils.py	                -> Common plotting functions
</details>

<details> <summary><b>notebooks/</b> – Jupyter notebooks for end-to-end workflow</summary>

notebooks/01_data_exploration.ipynb	

notebooks/02_target_processing.ipynb	

notebooks/03_predictor_processing.ipynb	

notebooks/04_model_training_classification.ipynb	

notebooks/05_model_training_regression.ipynb	

notebooks/06_model_comparison_ttest.ipynb	

notebooks/07_feature_importance.ipynb	
</details>


requirements.txt	            -> Python dependencies

README.md	                    -> Project documentation (this file)

## Run code

### 1. Install Environment
```bash

git clone https://github.com/angelinedorvil/US-States-Happiness-Ranking.git

cd US-States-Happiness-Ranking

pip install -r requirements.txt

python train_all_models.py *OR* use Notebooks (see below)
```

##  Datasets

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