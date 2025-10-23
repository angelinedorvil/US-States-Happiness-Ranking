from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from config import RANDOM_STATE
import joblib, json

def train_model(X_train, y_train, out_dir):

    """
    5. Optional Experiments (for bonus points or milestone 3)

        Try class_weight="balanced" to offset any class imbalance.

        Add max_features=["sqrt", 0.5, None] to your grid for feature-subspace tuning.

        Compare with DecisionTreeClassifier or XGBClassifier using the same composite index to 
        mirror the paper’s variety.

        Use PermutationImportance or feature_importances_ to identify which predictors drive your “well-being tier” 
        most strongly.
    """

    grid = {
        "n_estimators": [200, 400, 800], # number of trees
        "max_depth": [None, 5, 10], # maximum depth of each tree
        "min_samples_leaf": [1, 2, 4], # minimum samples required at each leaf node
    }

    # Initialize Random Forest Classifier and perform Grid Search with Cross-Validation
    model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)

    # Set up Stratified K-Fold Cross-Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # Perform Grid Search with F1 Macro scoring
    search = GridSearchCV(model, grid, cv=cv, scoring="f1_macro", n_jobs=-1, refit=True)

    # Fit the model
    search.fit(X_train, y_train)

    # Save the best model and CV results
    joblib.dump(search.best_estimator_, f"{out_dir}/RandomForest_best.joblib")

    # Save CV results to JSON
    with open(f"{out_dir}/RandomForest_cv.json", "w") as f:
        json.dump({"best_params": search.best_params_, "cv_f1_macro": float(search.best_score_)}, f, indent=2)

    # Return the best estimator
    return search.best_estimator_
