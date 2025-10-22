from src.models import model_random_forest, model_decision_tree, model_knn, model_svm
from src.utils import evaluate_model
from src.config import RESULTS_DIR
import pandas as pd, joblib
from sklearn.model_selection import train_test_split
from src.config import RANDOM_STATE

# Load your merged dataset
df = pd.read_csv("results/norm_predictors/predictor_dataset_combined.csv").merge(
    pd.read_csv("results/norm_targets/env_safety_index_by_state.csv")[["State","Percentile_Class"]],
    on="State", how="inner"
)

features = [c for c in df.columns if c.endswith("_norm")]
X, y = df[features], df["Percentile_Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    stratify=y, random_state=RANDOM_STATE)

rf = model_random_forest.train_model(X_train, y_train, RESULTS_DIR)
dt = model_decision_tree.train_model(X_train, y_train, RESULTS_DIR)
knn = model_knn.train_model(X_train, y_train, RESULTS_DIR)
svm = model_svm.train_model(X_train, y_train, RESULTS_DIR)

for name, model in {"RandomForest": rf, "DecisionTree": dt, "kNN": knn, "SVM": svm}.items():
    evaluate_model(model, X_test, y_test, name, RESULTS_DIR)
