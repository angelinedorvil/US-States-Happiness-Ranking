from models import model_decision_tree_class, model_knn_class, model_random_forest_class, model_svm_class, model_random_forest_regress, model_decision_tree_regress, model_knn_regress, model_svm_regress, model_xgboost_regress, model_ridge_regress, model_lasso_regress, model_linear_regress, model_polyn_regress, model_mlp_regress
from utils.evaluation_utils import evaluate_model, evaluate_regression
from config import RESULTS_DIR, PLOTS_DIR, RANDOM_STATE
import pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from process_predictor_data import build_predictor_index
from process_target_data import build_target_index


# Train 9 Classifier models
# classifier_models = {
#     "RandomForest_classification": model_random_forest_class,
#     "kNN_classification": model_knn_class,
#     "DecisionTree_classification": model_decision_tree_class,
#     "SVM_classification": model_svm_class
# }

# Train 9 regressor models
regressor_models = {
    "RandomForest_regression": model_random_forest_regress,
    "kNN_regression": model_knn_regress, # Testing Extra
    "DecisionTree_regression": model_decision_tree_regress, 
    "SVM_regression": model_svm_regress,
    "XGBoost_regression": model_xgboost_regress,
    "Ridge_regression": model_ridge_regress,
    "Lasso_regression": model_lasso_regress,
    "Linear_regression": model_linear_regress,
    "Polynomial_regression": model_polyn_regress,
    "MLP_regression": model_mlp_regress

}

def classifier_initialization():
    # Load datasets
    df = pd.read_csv("results/norm_predictors/final_predictor_index_all_years.csv").merge(
        pd.read_csv("results/norm_targets/final_target_index_all_years.csv")[["State","Percentile_Class"]],
        on="State", how="inner"
    )

    # Retain normalized features only
    features = [c for c in df.columns if c.endswith("_norm")]

    # Split data
    X, y = df[features], df["Percentile_Class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        stratify=y, random_state=RANDOM_STATE)

    X_train, X_test = fill_in_missing_values(X_train, X_test)

    type = "classification"

    evaluate_and_save(classifier_models, X_test, y_test, X_train, y_train, RESULTS_DIR, PLOTS_DIR, type)

    return 

def regression_initialization():
    # Load datasets
    df = pd.read_csv("results/norm_predictors/final_predictor_index_all_years.csv").merge(
        pd.read_csv("results/norm_targets/final_target_index_all_years.csv")[["State","target_index"]],
        on="State", how="inner"
    )

    # Retain normalized features only
    features = [c for c in df.columns if c.endswith("_norm")]

    # Split data
    X, y = df[features], df["target_index"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=RANDOM_STATE)

    X_train, X_test = fill_in_missing_values(X_train, X_test)

    type = "regression"

    evaluate_and_save(regressor_models, X_test, y_test, X_train, y_train, RESULTS_DIR, PLOTS_DIR, type)

    return

def fill_in_missing_values(X_train, X_test):
    # Impute missing values using mean strategy
    imputer = SimpleImputer(strategy="mean")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    return X_train, X_test

def evaluate_and_save(models, X_test, y_test, X_train, y_train, model_directory, plot_directory, type):
    for model_name, model_module in models.items():
        model = model_module.train_model(X_train, y_train, model_directory)
        if type == "classification":
            evaluate_model(model, X_test, y_test, model_name, model_directory, plot_directory)
        else:
            evaluate_regression(model, X_test, y_test, model_name, model_directory, plot_directory)

if __name__ == "__main__":
    #classifier_initialization() For Milestone 3
    build_target_index()
    build_predictor_index()
    regression_initialization()