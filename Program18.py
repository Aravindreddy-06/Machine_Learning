import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target)
scaler = StandardScaler().fit(X_train)
X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

# ---------- Part A: Perceptron from scratch (setosa vs the rest) ----------
yb_train = (y_train == 0).astype(int)      # 1 = setosa, 0 = other
yb_test = (y_test == 0).astype(int)

w = np.zeros(4)
b = 0.0
lr = 0.1
for epoch in range(1, 11):
    errors = 0
    for x, t in zip(X_train, yb_train):
        pred = 1 if x @ w + b > 0 else 0          # step activation
        update = lr * (t - pred)                   # perceptron learning rule
        w += update * x
        b += update
        errors += int(update != 0)
    print(f"Epoch {epoch:>2}: mistakes = {errors}")
    if errors == 0:
        break

pred_test = (X_test @ w + b > 0).astype(int)
print("Setosa vs rest - test accuracy:", accuracy_score(yb_test, pred_test))

# ---------- Part B: all 3 species with scikit-learn's Perceptron ----------
model = Perceptron(max_iter=1000, random_state=42).fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\n3-class Perceptron accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

new_flower = [[5.1, 3.5, 1.4, 0.2]]
print("\nNew flower", new_flower[0], "->", iris.target_names[model.predict(scaler.transform(new_flower))[0]])