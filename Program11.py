import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Create a sample credit dataset (replace this with your own CSV if you have one)
rng = np.random.default_rng(42)
n = 2000
df = pd.DataFrame({
    "Annual_Income":        rng.normal(50000, 20000, n).clip(10000),
    "Num_Credit_Cards":     rng.integers(1, 10, n),
    "Interest_Rate":        rng.uniform(1, 30, n),
    "Num_Delayed_Payments": rng.integers(0, 25, n),
    "Credit_Utilization":   rng.uniform(10, 90, n),
    "Outstanding_Debt":     rng.uniform(0, 5000, n),
})

# Score from the features (+ noise), then split into Poor / Standard / Good
score = (df["Annual_Income"] / 10000 - df["Num_Delayed_Payments"] * 0.4
         - df["Interest_Rate"] * 0.15 - df["Credit_Utilization"] * 0.05
         - df["Outstanding_Debt"] / 1000 + rng.normal(0, 1, n))
df["Credit_Score"] = pd.qcut(score, [0, 0.3, 0.7, 1], labels=["Poor", "Standard", "Good"])

print(df.head(), "\n")
print(df["Credit_Score"].value_counts(), "\n")

# 2. Split into features (X) and label (y)
X = df.drop(columns="Credit_Score")
y = df["Credit_Score"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Train the classifier
model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
y_pred = model.predict(X_test)

# 4. Results
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
labels = ["Poor", "Standard", "Good"]
print("\nConfusion Matrix (rows = actual, cols = predicted):")
print(pd.DataFrame(confusion_matrix(y_test, y_pred, labels=labels), index=labels, columns=labels))
print("\n", classification_report(y_test, y_pred, labels=labels))

# 5. Which features matter most
print(pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).round(3))

# 6. Classify a new customer
new = pd.DataFrame([{"Annual_Income": 80000, "Num_Credit_Cards": 3, "Interest_Rate": 5,
                     "Num_Delayed_Payments": 1, "Credit_Utilization": 20, "Outstanding_Debt": 300}])
print("\nNew customer ->", model.predict(new)[0])