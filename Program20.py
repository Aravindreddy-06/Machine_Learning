import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Create a sample advertising dataset (replace with your own CSV if you have one)
rng = np.random.default_rng(42)
n = 200
df = pd.DataFrame({
    "TV":        rng.uniform(10, 300, n).round(1),     # ad spend in thousands
    "Radio":     rng.uniform(0, 50, n).round(1),
    "Newspaper": rng.uniform(0, 100, n).round(1),
})
df["Sales"] = (3 + 0.045 * df["TV"] + 0.19 * df["Radio"] + 0.0 * df["Newspaper"]
               + rng.normal(0, 1.5, n)).round(1)
print(df.head(), "\n")

# 2. Split data
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train Linear Regression
model = LinearRegression().fit(X_train, y_train)
pred = model.predict(X_test)
print(f"R2  = {r2_score(y_test, pred):.4f}")
print(f"MAE = {mean_absolute_error(y_test, pred):.3f}")

# 4. Effect of each advertising channel
print("\nSales increase per 1 unit more spend:")
print(pd.Series(model.coef_, index=X.columns).round(4))

# 5. Plot actual vs predicted
plt.scatter(y_test, pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
plt.xlabel("Actual sales"); plt.ylabel("Predicted sales"); plt.title("Sales Prediction")
plt.show()

# 6. Predict future sales for a planned budget
future = pd.DataFrame([{"TV": 200, "Radio": 30, "Newspaper": 20}])
print(f"\nPredicted sales for TV=200, Radio=30, Newspaper=20: {model.predict(future)[0]:.2f}")