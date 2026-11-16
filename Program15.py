import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

# 1. Load data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target)

# 2. Train and predict
nb = GaussianNB().fit(X_train, y_train)
y_pred = nb.predict(X_test)

# 3. Results
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n", classification_report(y_test, y_pred, target_names=iris.target_names))

ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=iris.target_names).plot(cmap="Blues")
plt.title("Naive Bayes - Iris")
plt.show()

# 4. Classify a new flower (with probabilities)
new_flower = [[6.0, 2.9, 4.5, 1.5]]   # sepal length, sepal width, petal length, petal width
probs = nb.predict_proba(new_flower)[0]
print("New flower", new_flower[0])
for name, p in zip(iris.target_names, probs):
    print(f"  P({name}) = {p:.3f}")
print("Prediction:", iris.target_names[probs.argmax()])