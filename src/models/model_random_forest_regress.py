# src/models/model_random_forest_regress.py
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold
from config import RANDOM_STATE
import joblib, json, os

def train_model(X_train, y_train, out_dir):
    """
    Trains a Random Forest Regressor with GridSearchCV
    and saves the best model + results.
    """

    # === Hyperparameter grid ===
    grid = {
        "n_estimators": [200, 400, 800],
        "max_depth": [None, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }

    # === Model and CV setup ===
    model = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    search = GridSearchCV(
        model,
        grid,
        cv=cv,
        scoring="r2",  # Regression metric
        n_jobs=-1,
        refit=True
    )

    # === Fit the model ===
    search.fit(X_train, y_train)

    # === Save best model ===
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(search.best_estimator_, f"{out_dir}/RandomForestRegress_best.joblib")

    # Get per-fold test scores for the best hyperparameters
    best_idx = search.best_index_
    split_keys = [k for k in search.cv_results_.keys() if k.startswith("split") and k.endswith("_test_score")]
    fold_scores = [float(search.cv_results_[k][best_idx]) for k in split_keys]

    # Save everything
    cv_summary = {
        "best_params": search.best_params_,
        "cv_r2_mean": float(search.best_score_),
        "cv_r2_per_fold": fold_scores
    }

    with open(f"{out_dir}/RandomForest_Regression_cv.json", "w") as f:
        json.dump(cv_summary, f, indent=2)

    return search.best_estimator_
