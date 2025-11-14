import numpy as np
import pandas as pd
from scipy.stats import mode
from dataclasses import dataclass
from typing import Optional
from collections import Counter

def read_data(trainfile='MNIST_train.csv', validationfile='MNIST_validation.csv'):
    dftrain = pd.read_csv(trainfile)
    dfval = pd.read_csv(validationfile)

    featurecols = list(dftrain.columns)
    featurecols.remove('label')
    featurecols.remove('even')
    targetcol = 'label'

    Xtrain = dftrain[featurecols].values / 256
    ytrain = dftrain[targetcol].values
    Xval = dfval[featurecols].values / 256
    yval = dfval[targetcol].values

    return Xtrain, ytrain, Xval, yval

class SoftmaxRegression:
    def __init__(self, epochs=100, learning_rate=0.1, batch_size=-1, random_state=None):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.random_state = random_state
        self.weights = None

    def softmax(self, z):
        z -= np.max(z, axis=1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(axis=-1, keepdims=True)

    def fit(self, X, y):
        np.random.seed(self.random_state)
        X = np.asarray(X, float)
        y = np.asarray(y)
        assert len(X) == len(y), f'Number of entries in X ({len(X)}) does not match that in y ({len(y)})'

        n, m = X.shape
        K = len(np.unique(y))

        X = np.c_[np.ones((n, 1)), X]
        Y = np.zeros((n, K))
        Y[np.arange(n), y] = 1
        self.weights = np.random.randn(m + 1, K)

        if self.batch_size == -1:
            self.batch_size = n

        for epoch in range(self.epochs):
            idx = np.random.permutation(n)
            X_shuffled = X[idx]
            y_shuffled = Y[idx]
            for i in range(int(n // self.batch_size)):
                X_batch = X_shuffled[i * self.batch_size : (i + 1) * self.batch_size]
                y_batch = y_shuffled[i * self.batch_size : (i + 1) * self.batch_size]
                y_hat = self.softmax(X_batch @ self.weights)
                self.weights -= self.learning_rate * (X_batch.T @ (y_hat - y_batch))

    def predict_proba(self, X):
        X = np.asarray(X, float)
        X = np.c_[np.ones((len(X), 1)), X]
        return self.softmax(X @ self.weights)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=-1)

@dataclass(slots=True)
class Node:
    feat_idx: int
    threshold: float
    left: "Node"
    right: "Node"

@dataclass(slots=True)
class Leaf:
    value: int

class DecisionTree:
    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.array(y)
        assert len(X) == len(y), f'Number of entries in X ({len(X)}) does not match that in y ({len(y)})'
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        num_classes = len(np.unique(y))
        num_samples = len(y)

        if (depth >= self.max_depth or
            num_classes == 1 or
            num_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Leaf(value=leaf_value)

        best_feat, best_thresh = self._best_split(X, y)
        if best_feat is None:
            leaf_value = self._most_common_label(y)
            return Leaf(value=leaf_value)

        left_idx = X[:, best_feat] <= best_thresh
        right_idx = X[:, best_feat] > best_thresh

        if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
            leaf_value = self._most_common_label(y)
            return Leaf(value=leaf_value)
            
        left = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right = self._build_tree(X[right_idx], y[right_idx], depth + 1)

        return Node(feat_idx=best_feat, threshold=best_thresh, left=left, right=right)

    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        if n_samples <= 1:
            return None, None

        best_gain = 0
        best_feat, best_thresh = None, None
        parent_gini = self._gini(y)

        classes, y_encoded = np.unique(y, return_inverse=True)
        n_classes = len(classes)

        for feat_idx in range(n_features):
            # Sort samples by this feature
            sort_idx = np.argsort(X[:, feat_idx])
            X_sorted = X[sort_idx, feat_idx]
            y_sorted = y_encoded[sort_idx]

            # Class counts cumulative (prefix sums)
            left_counts = np.zeros((n_samples, n_classes), dtype=int)
            np.add.at(left_counts, np.arange(n_samples), np.eye(n_classes, dtype=int)[y_sorted])
            left_counts = np.cumsum(left_counts, axis=0)

            total_counts = left_counts[-1]
            left_n = np.arange(1, n_samples + 1)
            right_n = n_samples - left_n

            # Avoid division by zero
            left_n[left_n == 0] = 1
            right_n[right_n == 0] = 1

            # Compute probabilities and Gini impurities
            left_prob = left_counts / left_n[:, None]
            right_prob = (total_counts - left_counts) / right_n[:, None]

            gini_left = 1.0 - np.sum(left_prob ** 2, axis=1)
            gini_right = 1.0 - np.sum(right_prob ** 2, axis=1)

            # Weighted Gini for each split
            weighted_gini = (left_n / n_samples) * gini_left + (right_n / n_samples) * gini_right
            gains = parent_gini - weighted_gini

            # Ignore invalid (duplicate feature values)
            mask = np.r_[True, np.diff(X_sorted) > 0]
            gains[~mask] = -np.inf

            # Get best threshold for this feature
            i_best = np.argmax(gains)
            gain = gains[i_best]
            if gain > best_gain:
                best_gain = gain
                best_feat = feat_idx
                best_thresh = (X_sorted[i_best] + X_sorted[i_best - 1]) / 2 if i_best > 0 else X_sorted[0]

        return best_feat, best_thresh

    def _gini(self, y):
        if len(y) == 0:
            return 0
        return 1 - ((np.bincount(y) / len(y)) ** 2).sum()

    def _most_common_label(self, y):
        return Counter(y).most_common(1)[0][0]

    def predict(self, X):
        X = np.asarray(X, float)
        return np.array([self._predict(inputs, self.root) for inputs in X])

    def _predict(self, inputs, node):
        if isinstance(node, Leaf):
            return node.value
        if inputs[node.feat_idx] <= node.threshold:
            return self._predict(inputs, node.left)
        else:
            return self._predict(inputs, node.right)

class RandomForest:
    def __init__(self, n_estimators=10, max_depth=3, min_samples_split=2, subsample=0.5, colsample=0.5, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.subsample = subsample
        self.colsample = colsample
        self.random_state = random_state
        self.feat_idx = []
        self.trees = []

    def fit(self, X, y):
        np.random.seed(self.random_state)
        X = np.asarray(X, float)
        y = np.array(y)
        assert len(X) == len(y), f'Number of entries in X ({len(X)}) does not match that in y ({len(y)})'
        
        n, m = X.shape
        for i in range(self.n_estimators):
            rows = np.random.choice(np.arange(n), int(self.subsample * n), replace=True)
            cols = np.random.choice(np.arange(m), int(self.colsample * m), replace=False)
            self.feat_idx.append(cols)
            X_sample, y_sample = X[np.ix_(rows, cols)], y[rows]
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        X = np.asarray(X, float)
        all_preds = np.vstack([
            tree.predict(X[:, cols])
            for tree, cols in zip(self.trees, self.feat_idx)
        ])
        y_pred, _ = mode(all_preds, axis=0, keepdims=False)
        return y_pred

class Perceptron:
    def __init__(self, epochs=100):
        self.epochs = epochs
        self.w = None
        self.b = None

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.array(y)
        assert len(X) == len(y), f'Number of entries in X ({len(X)}) does not match that in y ({len(y)})'

        y = np.where(y <= 0, -1, 1)
        self.w = np.zeros(X.shape[1])
        self.b = 0

        for epoch in range(self.epochs):
            pred = X @ self.w + self.b
            mask = y * pred <= 0
            if np.any(mask):
                self.w += np.sum(X[mask] * y[mask, np.newaxis], axis=0)
                self.b += np.sum(y[mask])
            else:
                print(f'Converged in {epoch} epochs. Stopping training')
                break

    def predict_score(self, X):
        X = np.asarray(X, float)
        return X @ self.w + self.b

    def predict(self, X):
        return np.where(self.predict_score(X) >= 0, 1, 0)

class SVM:
    def __init__(self, epochs=100, learning_rate=0.01, c=0.01):
        self.learning_rate = learning_rate
        self.c = c
        self.epochs = epochs
        self.w = None
        self.b = None

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.array(y)
        assert len(X) == len(y), f'Number of entries in X ({len(X)}) does not match that in y ({len(y)})'

        y = np.where(y <= 0, -1, 1)
        self.w = np.zeros(X.shape[1])
        self.b = 0

        for epoch in range(self.epochs):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (x_i @ self.w + self.b) >= 1
                if condition:
                    self.w -= self.learning_rate * (self.c * self.w)
                else:
                    self.w -= self.learning_rate * (self.c * self.w - x_i * y[idx])
                    self.b += self.learning_rate * y[idx]

    def predict_score(self, X):
        X = np.asarray(X, float)
        return X @ self.w + self.b

    def predict(self, X):
        return np.where(self.predict_score(X) >= 0.5, 1, 0)

class OneVsAll:
    def __init__(self, classifier, *args, **kwargs):
        self.classifier = classifier
        self.args = args
        self.kwargs = kwargs
        self.models = []
        self.classes = None

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.array(y)
        assert len(X) == len(y), f'Number of entries in X ({len(X)}) does not match that in y ({len(y)})'

        self.classes = np.unique(y)
        for i in self.classes:
            model = self.classifier(*self.args, **self.kwargs)
            model.fit(X, np.where(y == i, 1, 0))
            self.models.append(model)

    def predict_proba(self, X):
        X = np.asarray(X, float)
        if hasattr(self.classifier, 'predict_proba'):
            return np.vstack([model.predict_proba(X) for model in self.models])
        elif hasattr(self.classifier, 'predict_score'):
            preds = np.vstack([model.predict_score(X) for model in self.models])
            return (preds - preds.min(axis=1, keepdims=True)) / (preds.max(axis=1, keepdims=True) - preds.min(axis=1, keepdims=True))
    
    def predict(self, X):
        return self.classes[self.predict_proba(X).argmax(axis=0)]
