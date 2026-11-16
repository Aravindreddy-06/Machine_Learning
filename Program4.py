import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

class NeuralNetwork:
    def __init__(self, layer_sizes, lr=0.5, seed=42):
        rng = np.random.default_rng(seed)
        self.lr = lr
        self.W = [rng.normal(0, 1/np.sqrt(layer_sizes[i]),
                             (layer_sizes[i], layer_sizes[i+1]))
                  for i in range(len(layer_sizes) - 1)]
        self.b = [np.zeros((1, layer_sizes[i+1])) for i in range(len(layer_sizes) - 1)]
        self.loss_history = []

    def forward(self, X):
        acts = [X]
        for W, b in zip(self.W, self.b):
            acts.append(sigmoid(acts[-1] @ W + b))
        return acts

    def backward(self, acts, y):
        n = y.shape[0]
        out = acts[-1]
        delta = (out - y) * out * (1 - out)             # output-layer error term
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for l in reversed(range(len(self.W))):
            gW[l] = acts[l].T @ delta / n
            gb[l] = delta.sum(axis=0, keepdims=True) / n
            if l > 0:                                    # propagate error backwards
                delta = (delta @ self.W[l].T) * acts[l] * (1 - acts[l])
        return gW, gb

    def fit(self, X, y, epochs=1000, print_every=None):
        for e in range(1, epochs + 1):
            acts = self.forward(X)
            loss = 0.5 * np.mean(np.sum((acts[-1] - y) ** 2, axis=1))
            self.loss_history.append(loss)
            gW, gb = self.backward(acts, y)
            for l in range(len(self.W)):                 # gradient descent update
                self.W[l] -= self.lr * gW[l]
                self.b[l] -= self.lr * gb[l]
            if print_every and e % print_every == 0:
                print(f"Epoch {e:>5}  loss = {loss:.4f}")
        return self

    def predict_proba(self, X):
        return self.forward(X)[-1]

from sklearn.datasets import load_iris

iris = load_iris()
print("Features:", iris.feature_names)
print("Classes: ", iris.target_names.tolist())
print("Shape:   ", iris.data.shape)

pd.DataFrame(iris.data, columns=iris.feature_names).assign(species=iris.target_names[iris.target]).head()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

X_tr, X_te, y_tr, y_te = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target)

scaler = StandardScaler().fit(X_tr)
X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)
Y_tr = np.eye(3)[y_tr]          # one-hot targets: setosa=[1,0,0], versicolor=[0,1,0], virginica=[0,0,1]

print("Train:", X_tr.shape, "| Test:", X_te.shape)

net = NeuralNetwork([4, 8, 3], lr=0.5, seed=42).fit(X_tr, Y_tr, epochs=2000, print_every=500)

pred_tr = net.predict_proba(X_tr).argmax(axis=1)
pred_te = net.predict_proba(X_te).argmax(axis=1)
print(f"\nTrain accuracy: {accuracy_score(y_tr, pred_tr):.4f}")
print(f"Test accuracy:  {accuracy_score(y_te, pred_te):.4f}\n")
print("Confusion matrix (rows = actual, cols = predicted):")
print(confusion_matrix(y_te, pred_te))
print()
print(classification_report(y_te, pred_te, target_names=iris.target_names))

plt.plot(net.loss_history)
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Iris training loss")
plt.show()

samples = {"setosa-like":     [5.1, 3.5, 1.4, 0.2],
           "versicolor-like": [6.0, 2.9, 4.5, 1.5],
           "virginica-like":  [6.9, 3.1, 5.4, 2.1]}   # sepal L, sepal W, petal L, petal W

for name, s in samples.items():
    p = net.predict_proba(scaler.transform([s]))[0]
    print(f"{name:<16} {s} -> {iris.target_names[p.argmax()]}  (activations: {p.round(3)})")