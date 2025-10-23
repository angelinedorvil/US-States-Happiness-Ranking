# src/models/model_svm_regress.py
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from config import RANDOM_STATE
import joblib, json, os

def train_model(X_train, y_train, out_dir):
    # Define pipeline: scale first, then regress
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVR())
    ])

    grid = {
        "svm__kernel": ["linear", "poly", "rbf"],
        "svm__C": [0.1, 1, 10],
        "svm__degree": [2, 3],
        "svm__gamma": ["scale", "auto"]
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    search = GridSearchCV(pipeline, grid, cv=cv, scoring="r2", n_jobs=-1, refit=True)
    search.fit(X_train, y_train)

    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(search.best_estimator_, f"{out_dir}/SVMRegress_best.joblib")

    results = {
        "best_params": search.best_params_,
        "cv_r2": float(search.best_score_)
    }
    with open(f"{out_dir}/SVMRegress_cv.json", "w") as f:
        json.dump(results, f, indent=2)

    return search.best_estimator_
