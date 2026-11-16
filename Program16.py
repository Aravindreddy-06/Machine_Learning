import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Load data (0 = malignant, 1 = benign)
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42, stratify=data.target)

# 2. Algorithms to compare (scaling included for the ones that need it)
models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "K-Nearest Neighbours": make_pipeline(StandardScaler(), KNeighborsClassifier(5)),
    "Naive Bayes":          GaussianNB(),
    "Decision Tree":        DecisionTreeClassifier(random_state=42),
    "Random Forest":        RandomForestClassifier(random_state=42),
    "SVM":                  make_pipeline(StandardScaler(), SVC()),
}

# 3. Train each model and evaluate
rows = []
for name, model in models.items():
    cv = cross_val_score(model, X_train, y_train, cv=5).mean()      # 5-fold CV on training data
    pred = model.fit(X_train, y_train).predict(X_test)
    rows.append([name, cv, accuracy_score(y_test, pred), precision_score(y_test, pred),
                 recall_score(y_test, pred), f1_score(y_test, pred)])

results = pd.DataFrame(rows, columns=["Model", "CV Acc", "Test Acc", "Precision", "Recall", "F1"])
results = results.sort_values("Test Acc", ascending=False).round(4)
print(results.to_string(index=False))

# 4. Bar chart
results.plot(x="Model", y=["CV Acc", "Test Acc"], kind="barh", figsize=(8, 4), xlim=(0.85, 1))
plt.title("Classifier comparison"); plt.tight_layout(); plt.show()