from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from config import RANDOM_STATE
import joblib, json

def train_model(X_train, y_train, out_dir):
    """
    Train a Support Vector Machine classifier using a pipeline with StandardScaler.
    Performs grid search with stratified 5-fold CV.
    """

    # === SVM Pipeline (scaling + model) ===
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(probability=True, random_state=RANDOM_STATE))
    ])

    # === Parameter grid ===
    grid = {
        "svm__kernel": ["linear", "rbf", "poly"],
        "svm__C": [0.1, 1, 10, 100],
        "svm__gamma": ["scale", "auto"],
        "svm__degree": [2, 3],  # Only relevant for 'poly'
    }

    # === Cross-validation setup ===
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # === Grid Search ===
    search = GridSearchCV(
        pipe,
        grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        refit=True
    )

    # === Fit ===
    search.fit(X_train, y_train)

    # === Save the best model ===
    joblib.dump(search.best_estimator_, f"{out_dir}/SVM_best.joblib")

    # === Save the cross-validation results ===
    with open(f"{out_dir}/SVM_cv.json", "w") as f:
        json.dump(
            {
                "best_params": search.best_params_,
                "cv_f1_macro": float(search.best_score_)
            },
            f,
            indent=2
        )

    return search.best_estimator_
