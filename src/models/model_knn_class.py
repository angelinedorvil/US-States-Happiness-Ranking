from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from config import RANDOM_STATE
import joblib, json

def train_model(X_train, y_train, out_dir):
    """
    Train a k-Nearest Neighbors classifier with GridSearchCV and Stratified 5-Fold CV.
    Saves the best model and cross-validation results.

    Try slightly larger k: [5, 7, 9, 11, 15] — can smooth over noise.

    Add StandardScaler

    Visualize confusion matrix heatmap

    """

    # --- Hyperparameter grid ---
    grid = {
        "n_neighbors": [3, 5, 7, 11],      # number of neighbors
        "weights": ["uniform", "distance"], # uniform = equal, distance = weighted by inverse distance
        "p": [1, 2]                         # 1 = Manhattan, 2 = Euclidean
    }

    # --- Initialize model ---
    model = KNeighborsClassifier(n_jobs=-1)

    # --- Cross-validation setup ---
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # --- Grid search ---
    search = GridSearchCV(
        estimator=model,
        param_grid=grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        refit=True
    )

    # --- Fit model ---
    search.fit(X_train, y_train)

    # --- Save best model ---
    joblib.dump(search.best_estimator_, f"{out_dir}/kNN_best.joblib")

    # --- Save CV results ---
    with open(f"{out_dir}/kNN_cv.json", "w") as f:
        json.dump({
            "best_params": search.best_params_,
            "cv_f1_macro": float(search.best_score_)
        }, f, indent=2)

    return search.best_estimator_
