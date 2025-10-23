from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from config import RANDOM_STATE
import joblib, json

def train_model(X_train, y_train, out_dir):
    """
    Train a Decision Tree Classifier with Grid Search Cross-Validation.
    Saves the best model and cross-validation results.
    Ovefits easily
    """

    # === Hyperparameter grid ===
    grid = {
        "criterion": ["gini", "entropy", "log_loss"],
        "max_depth": [5, 8, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4, 8],

        # "max_depth": [None, 5, 10, 20],
        # "min_samples_split": [2, 5, 10],
        # "min_samples_leaf": [1, 2, 4],
    }

    # === Initialize model ===
    model = DecisionTreeClassifier(random_state=RANDOM_STATE)

    # === Cross-validation setup ===
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # === Grid search ===
    search = GridSearchCV(
        model,
        grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        refit=True
    )

    # === Fit model ===
    search.fit(X_train, y_train)

    # === Save best model ===
    joblib.dump(search.best_estimator_, f"{out_dir}/DecisionTree_best.joblib")

    # === Save cross-validation results ===
    with open(f"{out_dir}/DecisionTree_cv.json", "w") as f:
        json.dump(
            {
                "best_params": search.best_params_,
                "cv_f1_macro": float(search.best_score_)
            },
            f,
            indent=2
        )

    return search.best_estimator_
