import pandas as pd
from google.colab import drive

drive.mount('/content/drive')

# Change this to wherever you saved the file in Drive
file_path = '/content/drive/MyDrive/play_tennis_dataset.csv'

# keep_default_na=False -> the text "None" stays a category instead of becoming NaN
df = pd.read_csv(file_path, keep_default_na=False).drop(columns="Day")

features = ["Outlook", "Temperature", "Humidity", "Wind"]
target = "Play"

print(df.shape)
df.head()

from math import log2
from collections import Counter

def entropy(col):
    n = len(col)
    return -sum((c/n) * log2(c/n) for c in Counter(col).values())

def info_gain(df, attr, target):
    remainder = sum(len(s)/len(df) * entropy(s[target]) for _, s in df.groupby(attr))
    return entropy(df[target]) - remainder

def id3(df, attrs, target="Play", depth=0, max_depth=None, verbose=True):
    majority = df[target].mode()[0]

    if df[target].nunique() == 1:                         # pure node -> leaf
        return df[target].iloc[0]
    if not attrs or (max_depth is not None and depth >= max_depth):
        return majority                                   # stop -> majority leaf

    gains = {a: info_gain(df, a, target) for a in attrs}
    best = max(gains, key=gains.get)
    if verbose:
        print("  " * depth + f"Split on {best} (n={len(df)}, gain={gains[best]:.4f})")

    tree = {best: {"__default__": majority}}              # fallback for unseen values
    for val, subset in df.groupby(best):
        tree[best][val] = id3(subset, [a for a in attrs if a != best],
                              target, depth + 1, max_depth, verbose)
    return tree

def classify(tree, sample):
    if not isinstance(tree, dict):
        return tree
    attr = next(iter(tree))
    branches = tree[attr]
    return classify(branches.get(sample[attr], branches["__default__"]), sample)

def print_tree(tree, indent=""):
    if not isinstance(tree, dict):
        print(indent + "→", tree)
        return
    attr = next(iter(tree))
    for val, sub in tree[attr].items():
        if val == "__default__":
            continue
        print(f"{indent}[{attr} = {val}]")
        print_tree(sub, indent + "    ")

print(f"Entropy of dataset: {entropy(df[target]):.4f}\n")

# 80/20 train-test split
train = df.sample(frac=0.8, random_state=42)
test  = df.drop(train.index)

# max_depth=3 keeps the printed tree readable; set to None for the full ID3 tree
tree = id3(train, features, target, max_depth=3)

print("\n--- Decision tree ---")
print_tree(tree)

# Accuracy on unseen data
preds = test.apply(lambda row: classify(tree, row), axis=1)
print(f"\nTest accuracy: {(preds == test[target]).mean():.4f}")

# Classify a new sample
new_sample = {"Outlook": "Sunny", "Temperature": "Cool",
              "Humidity": "High", "Wind": "Strong"}
print("\nNew sample:", new_sample)
print("Prediction: Play =", classify(tree, new_sample))