from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from config import RANDOM_STATE
import json, joblib, os
import numpy as np

def train_model(X_train, y_train, out_dir):
    """Train plain Linear Regression with 5-Fold CV."""

    model = LinearRegression()

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2")
    model.fit(X_train, y_train)

    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(model, f"{out_dir}/LinearRegress_best.joblib")

    cv_summary = {
        "best_params": {},
        "cv_r2_mean": float(np.mean(scores)),
        "cv_r2_per_fold": scores.tolist(),
    }
    with open(f"{out_dir}/Linear_Regression_cv.json", "w") as f:
        json.dump(cv_summary, f, indent=2)

    return model
