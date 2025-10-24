from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from config import RANDOM_STATE
import json, joblib, os

def train_model(X_train, y_train, out_dir):
    """Train Ridge Regression with 5-Fold CV."""

    grid = {"alpha": [0.01, 0.1, 1, 10, 100]}

    model = Ridge(random_state=RANDOM_STATE)

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

    search.fit(X_train, y_train)

    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(search.best_estimator_, f"{out_dir}/RidgeRegress_best.joblib")

    best_idx = search.best_index_
    split_keys = [k for k in search.cv_results_.keys() if k.startswith("split") and k.endswith("_test_score")]
    fold_scores = [float(search.cv_results_[k][best_idx]) for k in split_keys]

    cv_summary = {
        "best_params": search.best_params_,
        "cv_r2_mean": float(search.best_score_),
        "cv_r2_per_fold": fold_scores,
    }
    with open(f"{out_dir}/Ridge_Regression_cv.json", "w") as f:
        json.dump(cv_summary, f, indent=2)

    return search.best_estimator_
