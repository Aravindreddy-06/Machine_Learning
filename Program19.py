import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Create a sample bank loan dataset (replace with your own CSV if you have one)
rng = np.random.default_rng(42)
n = 1500
df = pd.DataFrame({
    "Age":            rng.integers(21, 65, n),
    "Income":         rng.normal(50000, 20000, n).clip(12000).round(),
    "Loan_Amount":    rng.uniform(5000, 100000, n).round(),
    "Credit_History": rng.choice([0, 1], n, p=[0.25, 0.75]),     # 1 = good history
    "Employed":       rng.choice([0, 1], n, p=[0.2, 0.8]),
})
score = (df["Credit_History"] * 2.5 + df["Employed"] * 1.0
         + (df["Income"] - df["Loan_Amount"]) / 30000 + rng.normal(0, 1, n))
df["Loan_Approved"] = (score > 1.5).astype(int)
print(df.head(), "\n")
print("Approval rate:", df["Loan_Approved"].mean().round(3), "\n")

# 2. Split data
X = df.drop(columns="Loan_Approved")
y = df["Loan_Approved"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Train Naive Bayes
nb = GaussianNB().fit(X_train, y_train)
y_pred = nb.predict(X_test)

# 4. Results
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nConfusion Matrix (rows = actual, cols = predicted):\n", confusion_matrix(y_test, y_pred))
print("\n", classification_report(y_test, y_pred, target_names=["Rejected", "Approved"]))

# 5. Predict for a new applicant
applicant = pd.DataFrame([{"Age": 35, "Income": 60000, "Loan_Amount": 25000,
                           "Credit_History": 1, "Employed": 1}])
p = nb.predict_proba(applicant)[0]
print(f"New applicant -> P(Rejected) = {p[0]:.3f}, P(Approved) = {p[1]:.3f}")
print("Decision:", "Approved" if p[1] > 0.5 else "Rejected")