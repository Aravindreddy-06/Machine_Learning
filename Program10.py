import numpy as np
import matplotlib.pyplot as plt

# 1. Create data from two Gaussians (the "true" values we want EM to recover)
rng = np.random.default_rng(0)
data = np.concatenate([rng.normal(2, 1.0, 300),     # group 1: mean 2, std 1
                       rng.normal(8, 1.5, 200)])    # group 2: mean 8, std 1.5

def gaussian(x, mean, std):
    return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))

# 2. Initial guess for the parameters
means = np.array([1.0, 5.0])
stds = np.array([1.0, 1.0])
weights = np.array([0.5, 0.5])

# 3. EM algorithm
for step in range(1, 51):
    # E-step: probability that each point belongs to each group
    resp = np.array([weights[k] * gaussian(data, means[k], stds[k]) for k in range(2)])
    resp /= resp.sum(axis=0)

    # M-step: re-estimate parameters using those probabilities
    Nk = resp.sum(axis=1)
    means = (resp * data).sum(axis=1) / Nk
    stds = np.sqrt((resp * (data - means[:, None]) ** 2).sum(axis=1) / Nk)
    weights = Nk / len(data)

    if step in (1, 2, 5, 10, 50):
        print(f"Step {step:>2}: means = {means.round(3)}, stds = {stds.round(3)}, weights = {weights.round(3)}")

# 4. Final result
print("\nTrue values : means = [2, 8], stds = [1, 1.5], weights = [0.6, 0.4]")
print("EM estimate :", f"means = {means.round(2)}, stds = {stds.round(2)}, weights = {weights.round(2)}")

# 5. Plot
x = np.linspace(-2, 14, 300)
plt.hist(data, bins=30, density=True, alpha=0.4, color="gray")
plt.plot(x, sum(weights[k] * gaussian(x, means[k], stds[k]) for k in range(2)), "r-", label="EM fit")
plt.legend(); plt.title("Expectation-Maximization (2 Gaussians)")
plt.show()

# 6. Check against scikit-learn
from sklearn.mixture import GaussianMixture
gm = GaussianMixture(2, random_state=0).fit(data.reshape(-1, 1))
print("\nscikit-learn means:", np.sort(gm.means_.ravel()).round(2))