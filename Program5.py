import numpy as np
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 1. Load data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42)

# 2. K-NN function
def knn_predict(x, X_train, y_train, k=5):
    distances = np.sqrt(((X_train - x) ** 2).sum(axis=1))   # distance to every training point
    nearest = np.argsort(distances)[:k]                      # indices of k closest points
    votes = Counter(y_train[nearest])                        # count their labels
    return votes.most_common(1)[0][0]                        # majority label

# 3. Test accuracy
k = 5
predictions = [knn_predict(x, X_train, y_train, k) for x in X_test]
accuracy = np.mean(np.array(predictions) == y_test)
print(f"k = {k}, Accuracy = {accuracy:.4f}")

# 4. Classify a new sample
new_sample = np.array([5.1, 3.5, 1.4, 0.2])
label = knn_predict(new_sample, X_train, y_train, k)
print("New sample", new_sample.tolist(), "->", iris.target_names[label])