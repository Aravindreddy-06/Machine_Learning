import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load data (predict disease progression from 10 features)
data = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42)

scaler = StandardScaler().fit(X_train)
X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

# 2. Linear Regression from scratch (gradient descent)
class LinearRegression:
    def __init__(self, lr=0.5, epochs=5000):
        self.lr, self.epochs = lr, epochs

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.b = 0.0
        for _ in range(self.epochs):
            error = (X @ self.w + self.b) - y              # prediction - actual
            self.w -= self.lr * (X.T @ error) / len(y)     # gradient descent step
            self.b -= self.lr * error.mean()
        return self

    def predict(self, X):
        return X @ self.w + self.b

# 3. Train and evaluate
model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

print("MSE :", round(mean_squared_error(y_test, y_pred), 2))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, y_pred)), 2))
print("R2  :", round(r2_score(y_test, y_pred), 4))

# 4. Plot actual vs predicted
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual"); plt.ylabel("Predicted"); plt.title("Linear Regression")
plt.show()

# 5. Check against scikit-learn
from sklearn.linear_model import LinearRegression as SkLR
sk = SkLR().fit(X_train, y_train)
print("\nscikit-learn R2:", round(sk.score(X_test, y_test), 4))
print("Same weights:", np.allclose(model.w, sk.coef_, atol=1e-3))