import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix
)

# --------------------------------------------------
# 1. Load data
# --------------------------------------------------
df = pd.read_csv("data/processed/segmented_data.csv")

# --------------------------------------------------
# 2. Define target and features
# --------------------------------------------------
y = df["Churned"]

X = df.drop(columns=[
    "Churned",          # target
    "segment",          # numeric cluster id (redundant)
])

# --------------------------------------------------
# 3. Encode categorical features
# --------------------------------------------------
X = pd.get_dummies(X, drop_first=True)

# --------------------------------------------------
# 4. Train-test split (stratified!)
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# --------------------------------------------------
# 5. Train improved Random Forest
# --------------------------------------------------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=1
)

model.fit(X_train, y_train)

# --------------------------------------------------
# 6. Predictions
# --------------------------------------------------
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# --------------------------------------------------
# 7. Evaluation
# --------------------------------------------------
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_proba)
print(f"ROC-AUC Score: {roc_auc:.3f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save classification report
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

report_df.to_csv("results/classification_report.csv")

# Save ROC-AUC
with open("results/roc_auc.txt", "w") as f:
    f.write(f"ROC-AUC Score: {roc_auc:.3f}")

#Save confusion matrix
cm = confusion_matrix(y_test, y_pred)

cm_df = pd.DataFrame(
    cm,
    index=["Actual_No_Churn", "Actual_Churn"],
    columns=["Pred_No_Churn", "Pred_Churn"]
)

cm_df.to_csv("results/confusion_matrix.csv")



# --------------------------------------------------
# 8. Feature importance (top drivers of churn)
# --------------------------------------------------
feature_importance = (
    pd.Series(model.feature_importances_, index=X.columns)
    .sort_values(ascending=False)
)
# Save feature importance
feature_importance.to_csv(
    "results/feature_importance.csv",
    header=["importance"]
)


print("\nTop 10 Churn Drivers:")
print(feature_importance.head(10))

# --------------------------------------------------
# 9. Save churn probabilities for business use
# --------------------------------------------------
df["churn_probability"] = model.predict_proba(X)[:, 1]

df.to_csv("data/processed/churn_predictions.csv", index=False)

print("\nChurn prediction completed and saved.")
