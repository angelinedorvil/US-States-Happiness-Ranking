# src/model_randomforest.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import os

# === Load preprocessed dataset ===
# Assuming data_load.py saved your final dataset
df = pd.read_csv("data/final_dataset.csv")

# --- Prepare features & target ---
# Ensure we have only numeric features
X = df[['life_expectancy', 'poverty_rate_2023']]

# Create 3 classes (Low, Medium, High) from mental_health_index or life_satisfaction_percent
if 'mental_health_index' in df.columns:
    y = pd.qcut(df['mental_health_index'], q=3, labels=['Low', 'Medium', 'High'])
else:
    y = pd.qcut(df['life_satisfaction_percent'], q=3, labels=['Low', 'Medium', 'High'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# === Train Random Forest ===
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# === Train Decision Tree ===
dt = DecisionTreeClassifier(random_state=42, max_depth=5)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

# === Evaluate both models ===
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
print("Decision Tree Accuracy:", accuracy_score(y_test, dt_pred))

print("\nRandom Forest Report:\n", classification_report(y_test, rf_pred))
print("\nDecision Tree Report:\n", classification_report(y_test, dt_pred))

# --- Save metrics for milestone submission ---
os.makedirs("results", exist_ok=True)
with open("results/metrics_rf_dt.txt", "w") as f:
    f.write("Random Forest:\n")
    f.write(classification_report(y_test, rf_pred))
    f.write("\n\nDecision Tree:\n")
    f.write(classification_report(y_test, dt_pred))

# --- Plot feature importances ---
importances = pd.Series(rf.feature_importances_, index=X.columns)
importances.sort_values().plot(kind="barh", title="Feature Importance (Random Forest)")
plt.tight_layout()
plt.savefig("results/feature_importance_rf.png")
plt.show()
