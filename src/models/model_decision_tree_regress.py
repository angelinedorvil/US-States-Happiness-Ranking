# src/models/model_decision_tree_regress.py
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, KFold
from config import RANDOM_STATE
import joblib, json, os

def train_model(X_train, y_train, out_dir):
    grid = {
        "criterion": ["squared_error", "friedman_mse", "absolute_error"],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }

    model = DecisionTreeRegressor(random_state=RANDOM_STATE)
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    search = GridSearchCV(model, grid, cv=cv, scoring="r2", n_jobs=-1, refit=True)
    search.fit(X_train, y_train)

    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(search.best_estimator_, f"{out_dir}/DecisionTreeRegress_best.joblib")

    results = {
        "best_params": search.best_params_,
        "cv_r2": float(search.best_score_)
    }
    with open(f"{out_dir}/DecisionTreeRegress_cv.json", "w") as f:
        json.dump(results, f, indent=2)

    return search.best_estimator_
