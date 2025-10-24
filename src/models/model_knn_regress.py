# src/models/model_knn_regress.py
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV, KFold
from config import RANDOM_STATE
import joblib, json, os

def train_model(X_train, y_train, out_dir):
    grid = {
        "n_neighbors": [3, 5, 7, 9],
        "weights": ["uniform", "distance"],
        "p": [1, 2]  # 1 = Manhattan, 2 = Euclidean
    }

    model = KNeighborsRegressor()
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    search = GridSearchCV(model, grid, cv=cv, scoring="r2", n_jobs=-1, refit=True)
    search.fit(X_train, y_train)

    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(search.best_estimator_, f"{out_dir}/KNNRegress_best.joblib")

    best_idx = search.best_index_
    split_keys = [k for k in search.cv_results_.keys() if k.startswith("split") and k.endswith("_test_score")]
    fold_scores = [float(search.cv_results_[k][best_idx]) for k in split_keys]

    results = {
        "best_params": search.best_params_,
        "cv_r2_mean": float(search.best_score_),
        "cv_r2_per_fold": fold_scores,
    }
    with open(f"{out_dir}/KNN_Regression_cv.json", "w") as f:
        json.dump(results, f, indent=2)

    return search.best_estimator_
