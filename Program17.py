import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Create a sample mobile dataset (replace with your own CSV if you have one)
rng = np.random.default_rng(42)
n = 1000
df = pd.DataFrame({
    "RAM_GB":        rng.choice([2, 3, 4, 6, 8, 12], n),
    "Storage_GB":    rng.choice([32, 64, 128, 256, 512], n),
    "Battery_mAh":   rng.integers(3000, 6000, n),
    "Camera_MP":     rng.choice([8, 12, 16, 48, 64, 108], n),
    "Screen_inches": rng.uniform(5.0, 7.0, n).round(1),
    "Has_5G":        rng.integers(0, 2, n),
})
df["Price"] = (df["RAM_GB"] * 1200 + df["Storage_GB"] * 40 + df["Battery_mAh"] * 1.5
               + df["Camera_MP"] * 90 + df["Screen_inches"] * 800 + df["Has_5G"] * 3000
               + rng.normal(0, 1500, n)).round()
print(df.head(), "\n")

# 2. Split data
X = df.drop(columns="Price")
y = df["Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train two models and compare
models = {"Linear Regression": LinearRegression(),
          "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)}
for name, m in models.items():
    pred = m.fit(X_train, y_train).predict(X_test)
    print(f"{name:<18} R2 = {r2_score(y_test, pred):.4f}   MAE = {mean_absolute_error(y_test, pred):,.0f}")

# 4. Plot actual vs predicted
lin = models["Linear Regression"]
plt.scatter(y_test, lin.predict(X_test), s=10)
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
plt.xlabel("Actual price"); plt.ylabel("Predicted price"); plt.title("Mobile Price Prediction")
plt.show()

# 5. Predict the price of a new phone
new_phone = pd.DataFrame([{"RAM_GB": 8, "Storage_GB": 128, "Battery_mAh": 5000,
                           "Camera_MP": 64, "Screen_inches": 6.5, "Has_5G": 1}])
print(f"\nPredicted price: {lin.predict(new_phone)[0]:,.0f}")