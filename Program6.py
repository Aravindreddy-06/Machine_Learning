import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay

# 1. Load data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target)

# 2. Gaussian Naive Bayes from scratch
class NaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.priors, self.means, self.vars = {}, {}, {}
        for c in self.classes:
            Xc = X[y == c]
            self.priors[c] = len(Xc) / len(X)          # P(class)
            self.means[c] = Xc.mean(axis=0)            # mean of each feature
            self.vars[c] = Xc.var(axis=0) + 1e-9       # variance of each feature
        return self

    def _log_posterior(self, x, c):
        # log P(class) + sum of log P(feature | class), using the Gaussian formula
        log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * self.vars[c])
                                       + (x - self.means[c]) ** 2 / self.vars[c])
        return np.log(self.priors[c]) + log_likelihood

    def predict(self, X):
        return np.array([max(self.classes, key=lambda c: self._log_posterior(x, c))
                         for x in X])

# 3. Train and predict
model = NaiveBayes().fit(X_train, y_train)
y_pred = model.predict(X_test)

# 4. Results
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred),
                       display_labels=iris.target_names).plot(cmap="Blues")
plt.title("Naive Bayes - Confusion Matrix")
plt.show()

# 5. Check against scikit-learn
from sklearn.naive_bayes import GaussianNB
sk = GaussianNB().fit(X_train, y_train)
print("\nscikit-learn accuracy:", round(sk.score(X_test, y_test), 4),
      "| same predictions:", (sk.predict(X_test) == y_pred).all())