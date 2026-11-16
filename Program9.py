import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. Create curved (non-linear) data: y = 0.5x^2 - x + 3 + noise
rng = np.random.default_rng(42)
X = np.sort(rng.uniform(-3, 3, 100)).reshape(-1, 1)
y = 0.5 * X.ravel() ** 2 - X.ravel() + 3 + rng.normal(0, 0.5, 100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Fit Linear (degree 1) and Polynomial (degree 2) regression
linear = LinearRegression().fit(X_train, y_train)
poly = make_pipeline(PolynomialFeatures(degree=2), LinearRegression()).fit(X_train, y_train)

# 3. Compare on the test set
for name, model in [("Linear", linear), ("Polynomial (deg 2)", poly)]:
    pred = model.predict(X_test)
    print(f"{name:<20} R2 = {r2_score(y_test, pred):.4f}   RMSE = {np.sqrt(mean_squared_error(y_test, pred)):.4f}")

# 4. Plot both fits
plt.scatter(X, y, color="gray", label="Data")
plt.plot(X, linear.predict(X), "r-", label="Linear")
plt.plot(X, poly.predict(X), "g-", label="Polynomial (deg 2)")
plt.legend(); plt.title("Linear vs Polynomial Regression")
plt.show()

# 5. Effect of the degree (too high = overfitting)
print("\nDegree   Train R2   Test R2")
for d in [1, 2, 5, 10]:
    m = make_pipeline(PolynomialFeatures(d), LinearRegression()).fit(X_train, y_train)
    print(f"{d:<8} {m.score(X_train, y_train):.4f}     {m.score(X_test, y_test):.4f}")