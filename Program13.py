import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Create a sample used-car dataset (replace with your own CSV if you have one)
rng = np.random.default_rng(42)
n = 1000
df = pd.DataFrame({
    "Year":          rng.integers(2005, 2023, n),
    "Present_Price": rng.uniform(3, 30, n).round(2),        # showroom price (lakhs)
    "Kms_Driven":    rng.integers(5000, 150000, n),
    "Fuel_Type":     rng.choice(["Petrol", "Diesel", "CNG"], n),
    "Owner":         rng.integers(0, 3, n),
})
age = 2023 - df["Year"]
df["Selling_Price"] = (df["Present_Price"] * (0.85 ** age) * (1 - df["Kms_Driven"] / 600000)
                       * np.where(df["Fuel_Type"] == "Diesel", 1.1, 1.0)
                       - df["Owner"] * 0.3 + rng.normal(0, 0.4, n)).clip(0.2).round(2)
print(df.head(), "\n")

# 2. Convert text column (Fuel_Type) to numbers
X = pd.get_dummies(df.drop(columns="Selling_Price"), columns=["Fuel_Type"], drop_first=True)
y = df["Selling_Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train two models and compare
models = {"Linear Regression": LinearRegression(),
          "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)}
for name, m in models.items():
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    print(f"{name:<18} R2 = {r2_score(y_test, pred):.4f}   MAE = {mean_absolute_error(y_test, pred):.3f} lakhs")

# 4. Plot actual vs predicted (Random Forest)
best = models["Random Forest"]
plt.scatter(y_test, best.predict(X_test))
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
plt.xlabel("Actual price"); plt.ylabel("Predicted price"); plt.title("Car Price Prediction")
plt.show()

# 5. Predict the price of a new car
new_car = pd.DataFrame([{"Year": 2018, "Present_Price": 10.0, "Kms_Driven": 40000, "Owner": 0,
                         "Fuel_Type_Diesel": 0, "Fuel_Type_Petrol": 1}])[X.columns]
print(f"\nPredicted selling price: {best.predict(new_car)[0]:.2f} lakhs")