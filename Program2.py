import pandas as pd
from itertools import product
from google.colab import drive

drive.mount('/content/drive')

# Change this to wherever you saved the file in Drive
file_path = '/content/drive/MyDrive/ENJOYSPORT.csv'

df = pd.read_csv(file_path)

features = ["Sky", "AirTemp", "Humidity", "Wind", "Water", "Forecast"]
target = "EnjoySport"
domains = {f: sorted(df[f].unique()) for f in features}   # allowed values per attribute

print(df.shape)
print(df, "\n")
print("Attribute values found in the data:")
for f in features:
    print(f"  {f}: {domains[f]}")

def covers(h, x):
    """True if hypothesis h classifies example x as positive."""
    return all(hv == "?" or (hv != "∅" and hv == xv) for hv, xv in zip(h, x))

def more_general_or_equal(h1, h2):
    """True if h1 is at least as general as h2."""
    if "∅" in h2:
        return True
    return all(a == "?" or a == b for a, b in zip(h1, h2))

def min_generalize(s, x):
    return tuple(xv if sv == "∅" else (sv if sv == xv else "?") for sv, xv in zip(s, x))

def min_specialize(g, x, domains, features):
    out = []
    for i, f in enumerate(features):
        if g[i] == "?":
            for v in domains[f]:
                if v != x[i]:
                    out.append(g[:i] + (v,) + g[i+1:])
    return out

def candidate_elimination(data, features, target, positive=1, verbose=True):
    n = len(features)
    S = [tuple(["∅"] * n)]
    G = [tuple(["?"] * n)]
    if verbose:
        print("Initial  S0 =", S, "\n         G0 =", G)
    for t, (_, row) in enumerate(data.iterrows(), 1):
        x = tuple(row[f] for f in features)
        if row[target] == positive:                       # positive example
            G = [g for g in G if covers(g, x)]
            newS = []
            for s in S:
                if covers(s, x):
                    newS.append(s)
                else:
                    h = min_generalize(s, x)
                    if any(more_general_or_equal(g, h) for g in G):
                        newS.append(h)
            S = [s for s in set(newS)
                 if not any(s != o and more_general_or_equal(s, o) for o in newS)]
        else:                                             # negative example
            S = [s for s in S if not covers(s, x)]
            newG = []
            for g in G:
                if not covers(g, x):
                    newG.append(g)
                else:
                    for h in min_specialize(g, x, domains, features):
                        if any(more_general_or_equal(h, s) for s in S):
                            newG.append(h)
            G = [g for g in set(newG)
                 if not any(g != o and more_general_or_equal(o, g) for o in newG)]
        S.sort(); G.sort()
        if verbose:
            label = "positive" if row[target] == positive else "negative"
            print(f"\nExample {t}: {list(x)} -> {label}")
            print(f"  S{t} =", [list(s) for s in S])
            print(f"  G{t} =", [list(g) for g in G])
    return S, G

S, G = candidate_elimination(df, features, target, positive=1)

print("\n" + "=" * 60)
print("Final specific boundary S:")
for s in S: print("  ", list(s))
print("Final general boundary G:")
for g in G: print("  ", list(g))

def version_space(S, G):
    vs = set()
    for s in S:
        idx = [i for i, v in enumerate(s) if v != "?"]
        for mask in product([True, False], repeat=len(idx)):
            h = list(s)
            for keep, i in zip(mask, idx):
                if not keep:
                    h[i] = "?"
            h = tuple(h)
            if any(more_general_or_equal(g, h) for g in G):
                vs.add(h)
    return sorted(vs, key=lambda h: (h.count("?"), h))

VS = version_space(S, G)
print(f"Version space: {len(VS)} hypotheses consistent with all training examples")
for h in VS: print("  ", list(h))

ok = all((covers(h, tuple(r[f] for f in features)) == (r[target] == 1))
         for h in VS for _, r in df.iterrows())
print("All verified consistent with training data:", ok)

def classify(sample):
    x = tuple(sample[f] for f in features)
    return sum(covers(h, x) for h in VS)

tests = [
 {"Sky":"Sunny","AirTemp":"Warm","Humidity":"Normal","Wind":"Strong","Water":"Cool","Forecast":"Change"},
 {"Sky":"Rainy","AirTemp":"Cold","Humidity":"Normal","Wind":"Weak","Water":"Warm","Forecast":"Same"},
 {"Sky":"Sunny","AirTemp":"Warm","Humidity":"Normal","Wind":"Weak","Water":"Warm","Forecast":"Same"},
 {"Sky":"Sunny","AirTemp":"Cold","Humidity":"Normal","Wind":"Strong","Water":"Warm","Forecast":"Same"},
]
for t in tests:
    v = classify(t)
    verdict = "Positive (1)" if v == len(VS) else "Negative (0)" if v == 0 else "Ambiguous"
    print(list(t.values()), f"-> {v}/{len(VS)} hypotheses say positive -> {verdict}")