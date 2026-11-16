import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Load data (binary: 0 = malignant, 1 = benign)
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42, stratify=data.target)

scaler = StandardScaler().fit(X_train)
X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

# 2. Logistic Regression from scratch
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class LogisticRegression:
    def __init__(self, lr=0.1, epochs=1000):
        self.lr, self.epochs = lr, epochs

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.b = 0.0
        for _ in range(self.epochs):
            p = sigmoid(X @ self.w + self.b)             # predicted probability
            self.w -= self.lr * (X.T @ (p - y)) / len(y) # gradient descent step
            self.b -= self.lr * (p - y).mean()
        return self

    def predict_proba(self, X):
        return sigmoid(X @ self.w + self.b)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

# 3. Train and evaluate
model = LogisticRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 4. Classify a new sample (first test row)
p = model.predict_proba(X_test[:1])[0]
print(f"\nP(benign) = {p:.3f} -> {data.target_names[int(p >= 0.5)]}  (actual: {data.target_names[y_test[0]]})")