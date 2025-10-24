# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier

# print("✅ pandas version:", pd.__version__)
# print("✅ scikit-learn test:", RandomForestClassifier)
import joblib
joblib_models = {
    "RandomForest_regression": "results/models/RandomForestRegress_best.joblib",
    "DecisionTree_regression": "results/models/DecisionTreeRegress_best.joblib",
    "kNN_regression": "results/models/kNNRegress_best.joblib",
    "SVM_regression": "results/models/SVMRegress_best.joblib"
}
for model_name, model_path in joblib_models.items():
    rf_model = joblib.load(model_path)
    print (f"rf_model for {model_name}", rf_model)