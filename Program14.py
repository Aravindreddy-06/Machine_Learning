import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Create a sample house dataset (replace with your own CSV if you have one)
rng = np.random.default_rng(42)
n = 1500
df = pd.DataFrame({
    "Area_sqft":      rng.uniform(500, 4000, n).round(),
    "Bedrooms":       rng.integers(1, 6, n),
    "Bathrooms":      rng.integers(1, 4, n),
    "Age_years":      rng.integers(0, 40, n),
    "Location_Score": rng.uniform(1, 10, n).round(1),   # 10 = best area
})
df["Price"] = (df["Area_sqft"] * 120 + df["Bedrooms"] * 15000 + df["Bathrooms"] * 10000
               - df["Age_years"] * 1500 + df["Location_Score"] * 25000
               + rng.normal(0, 20000, n)).round()
print(df.head(), "\n")

# 2. Split data
X = df.drop(columns="Price")
y = df["Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train two models and compare
models = {"Linear Regression": LinearRegression(),
          "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)}
for name, m in models.items():
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    print(f"{name:<18} R2 = {r2_score(y_test, pred):.4f}   MAE = {mean_absolute_error(y_test, pred):,.0f}")

# 4. Which features matter (Linear Regression coefficients)
lin = models["Linear Regression"]
print("\nPrice change per unit increase:")
print(pd.Series(lin.coef_, index=X.columns).round(0))

# 5. Plot actual vs predicted
plt.scatter(y_test, lin.predict(X_test), s=10)
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
plt.xlabel("Actual price"); plt.ylabel("Predicted price"); plt.title("House Price Prediction")
plt.show()

# 6. Predict the price of a new house
new_house = pd.DataFrame([{"Area_sqft": 1800, "Bedrooms": 3, "Bathrooms": 2,
                           "Age_years": 5, "Location_Score": 7.5}])
print(f"\nPredicted price: {lin.predict(new_house)[0]:,.0f}")