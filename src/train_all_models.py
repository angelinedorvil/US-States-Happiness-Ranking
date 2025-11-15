from models import model_random_forest_regress, model_decision_tree_regress, model_knn_regress, model_svm_regress, model_xgboost_regress, model_ridge_regress, model_lasso_regress, model_linear_regress, model_polyn_regress, model_mlp_regress
from utils.evaluation_utils import evaluate_regression
from utils.shap_utils import compute_and_plot_shap
from config import RESULTS_DIR, PLOTS_DIR, RANDOM_STATE
import pandas as pd, joblib, numpy as np, glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from process_predictor_data import build_predictor_index
from process_target_data import build_target_index
from pathlib import Path

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

def regression_initialization():
    # Load datasets
    df = pd.read_csv("results/norm_predictors/final_predictor_index_5_years.csv").merge(
        pd.read_csv("results/norm_targets/final_target_index_5_years.csv")[["State","target_index"]],
        on="State", how="inner"
    )

    # Retain normalized features only
    features = [c for c in df.columns if c.endswith("_norm")]

    # Split data
    X, y = df[features], df["target_index"]
    # Preserve names before imputing
    feature_names = X.columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=RANDOM_STATE)

    X_train, X_test = fill_in_missing_values(X_train, X_test)

    type = "regression"

    evaluate_and_save(regressor_models, X_test, y_test, X_train, y_train, RESULTS_DIR, PLOTS_DIR, type, feature_names)

    return

def fill_in_missing_values(X_train, X_test):
    # Impute missing values using mean strategy
    imputer = SimpleImputer(strategy="mean")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    return X_train, X_test

def evaluate_and_save(models, X_test, y_test, X_train, y_train, model_directory, plot_directory, type, feature_names):
    for model_name, model_module in models.items():
        model = model_module.train_model(X_train, y_train, model_directory)
        evaluate_regression(model, X_test, y_test, model_name, model_directory, plot_directory)
        if model_name in ["Ridge_regression", "Lasso_regression", "XGBoost_regression", "RandomForest_regression", "Linear_regression"]:
            X_df = pd.DataFrame(X_test, columns=feature_names).astype(float)
            print("X_df dtypes:\n", X_df.dtypes)
            print("Any object cols? ->", X_df.select_dtypes(include=["object"]).columns.tolist())
            print("Sample suspicious values:",
                X_df.apply(lambda s: s.astype(str).str.contains(r"\[|\]|E-").any()).to_dict())
            compute_and_plot_shap(model, X_df, model_name, plot_directory)

    files = glob.glob(str(plot_directory / "*_shap_summary.csv"))

    dfs = []
    for f in files:
        model = Path(f).stem.replace("_shap_summary", "")
        df = pd.read_csv(f).assign(model=model)
        dfs.append(df)
    all_df = pd.concat(dfs)

    top10 = (all_df.groupby("model", group_keys=False).apply(lambda x: x.sort_values("mean_abs_shap", ascending=False).head(10)))
    
    stability = (top10.groupby("feature")["model"].nunique()
                 .reset_index(name="n_models_in_top10")
                 .sort_values("n_models_in_top10", ascending=False))
    
    stability["pct_models_in_top5_5yrs"] = stability["n_models_in_top10"] / top10["model"].nunique() * 100

    stability.to_csv(plot_directory / "shap_stability_across_models_5yrs.csv", index=False)

    print(stability.head(20))

if __name__ == "__main__":
    build_target_index() 
    build_predictor_index()
    regression_initialization()