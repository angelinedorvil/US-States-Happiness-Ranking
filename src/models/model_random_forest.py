from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from src.config import RANDOM_STATE
import joblib, json

def train_model(X_train, y_train, out_dir):
    grid = {
        "n_estimators": [200, 400, 800],
        "max_depth": [None, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }
    model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(model, grid, cv=cv, scoring="f1_macro", n_jobs=-1, refit=True)
    search.fit(X_train, y_train)
    joblib.dump(search.best_estimator_, f"{out_dir}/RandomForest_best.joblib")
    with open(f"{out_dir}/RandomForest_cv.json", "w") as f:
        json.dump({"best_params": search.best_params_, "cv_f1_macro": float(search.best_score_)}, f, indent=2)
    return search.best_estimator_
