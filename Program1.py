import pandas as pd
from google.colab import drive

drive.mount('/content/drive')

# Change this to wherever you saved the file in Drive
file_path = '/content/drive/MyDrive/ENJOYSPORT.csv'

df = pd.read_csv(file_path)

features = ["Sky", "AirTemp", "Humidity", "Wind", "Water", "Forecast"]
target = "EnjoySport"

print(df.shape)
df

def find_s(data, features, target, positive=1, verbose=True):
    h = ["∅"] * len(features)                       # most specific hypothesis
    if verbose:
        print("Initial h:", h, "\n")
    for i, (_, row) in enumerate(data.iterrows(), 1):
        x = [row[f] for f in features]
        if row[target] != positive:                 # FIND-S ignores negative examples
            if verbose:
                print(f"Row {i}: {x} -> negative, ignored")
            continue
        old = h.copy()
        for j, value in enumerate(x):               # minimally generalise h to cover x
            if h[j] == "∅":
                h[j] = value
            elif h[j] != value:
                h[j] = "?"
        if verbose:
            print(f"Row {i}: {x} -> positive")
            print(f"   h: {old} -> {h}" + ("" if old != h else "   (no change)"))
    return h

def matches(h, x):
    return all(hv in ("?", xv) for hv, xv in zip(h, x))

def predict(h, sample, features, positive=1, negative=0):
    return positive if matches(h, [sample[f] for f in features]) else negative

h = find_s(df, features, target, positive=1)
print("\nMost specific hypothesis:", h)

# Consistency check
pos = df[df[target] == 1]; neg = df[df[target] == 0]
cov_pos = pos.apply(lambda r: matches(h, [r[f] for f in features]), axis=1).mean()
cov_neg = neg.apply(lambda r: matches(h, [r[f] for f in features]), axis=1).mean()
print(f"Positives covered: {cov_pos:.0%} | Negatives covered: {cov_neg:.0%}")

# Classify new samples
new_sample = {"Sky": "Sunny", "AirTemp": "Warm", "Humidity": "Low",
              "Wind": "Strong", "Water": "Cool", "Forecast": "Same"}
print("\nNew sample:", new_sample)
print("Prediction: EnjoySport =", predict(h, new_sample, features))

neg_sample = {"Sky": "Rainy", "AirTemp": "Cold", "Humidity": "High",
              "Wind": "Strong", "Water": "Warm", "Forecast": "Change"}
print("Prediction (Rainy/Cold sample): EnjoySport =", predict(h, neg_sample, features))