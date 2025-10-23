from models import model_random_forest, model_decision_tree, model_knn, model_svm
from utils.evaluation_utils import evaluate_model
from config import RESULTS_DIR
import pandas as pd, joblib
from sklearn.model_selection import train_test_split
from config import RANDOM_STATE
from sklearn.impute import SimpleImputer


# Load datasets
df = pd.read_csv("results/norm_predictors/final_predictor_index_all_years.csv").merge(
    pd.read_csv("results/norm_targets/final_target_index_all_years.csv")[["State","Percentile_Class"]],
    on="State", how="inner"
)

# Retain normalized features only
features = [c for c in df.columns if c.endswith("_norm")]

# Split data
X, y = df[features], df["Percentile_Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    stratify=y, random_state=RANDOM_STATE)

imputer = SimpleImputer(strategy="mean")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Train 4 models
rf = model_random_forest.train_model(X_train, y_train, RESULTS_DIR)
dt = model_decision_tree.train_model(X_train, y_train, RESULTS_DIR)
knn = model_knn.train_model(X_train, y_train, RESULTS_DIR)
svm = model_svm.train_model(X_train, y_train, RESULTS_DIR)

# Evaluate models , "DecisionTree": dt, "kNN": knn, "SVM": svm
for name, model in {"RandomForest": rf, "kNN": knn, "DecisionTree": dt, "SVM": svm}.items():
    evaluate_model(model, X_test, y_test, name, RESULTS_DIR)
