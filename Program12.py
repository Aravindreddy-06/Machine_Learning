import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target)

# 2. Scale features (K-NN uses distances, so scale matters)
scaler = StandardScaler().fit(X_train)
X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

# 3. Train and predict
knn = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
y_pred = knn.predict(X_test)

# 4. Results
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n", classification_report(y_test, y_pred, target_names=iris.target_names))

# 5. Try different values of k
print("k   Test accuracy")
for k in [1, 3, 5, 7, 9, 11]:
    acc = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train).score(X_test, y_test)
    print(f"{k:<3} {acc:.4f}")

# 6. Classify a new flower
new_flower = [[5.1, 3.5, 1.4, 0.2]]   # sepal length, sepal width, petal length, petal width
pred = knn.predict(scaler.transform(new_flower))[0]
print("\nNew flower", new_flower[0], "->", iris.target_names[pred])