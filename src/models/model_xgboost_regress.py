from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, KFold
from config import RANDOM_STATE
import json, joblib, os

def train_model(X_train, y_train, out_dir):
    """
    Train an XGBoost Regressor using GridSearchCV and 5-Fold Cross Validation.
    Mirrors the Janani et al. setup for regression models.
    """

    # === Parameter grid ===
    grid = {
        "n_estimators": [100, 300, 500],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "min_child_weight": [1, 3, 5],
    }

    model = XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="auto",
        eval_metric="rmse",
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    search = GridSearchCV(
        model,
        grid,
        cv=cv,
        scoring="r2",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    # === Fit ===
    search.fit(X_train, y_train)

    # === Save best model ===
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(search.best_estimator_, f"{out_dir}/XGBoostRegress_best.joblib")

    # Get per-fold results
    best_idx = search.best_index_
    split_keys = [k for k in search.cv_results_.keys() if k.startswith("split") and k.endswith("_test_score")]
    fold_scores = [float(search.cv_results_[k][best_idx]) for k in split_keys]

    # Save CV summary
    cv_summary = {
        "best_params": search.best_params_,
        "cv_r2_mean": float(search.best_score_),
        "cv_r2_per_fold": fold_scores,
    }
    with open(f"{out_dir}/XGBoost_Regression_cv.json", "w") as f:
        json.dump(cv_summary, f, indent=2)

    return search.best_estimator_
