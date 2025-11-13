import numpy as np
import pandas as pd
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
        best_gain = 0
        split_idx, split_thresh = None, None
        for feat_idx in range(X.shape[1]):
            thresholds = np.unique(X[:, feat_idx])
            for thresh in thresholds:
                gain = self._gini_gain(y, X[:, feat_idx], thresh)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = thresh
        return split_idx, split_thresh

    def _gini_gain(self, y, feature_column, threshold):
        left_idx = feature_column <= threshold
        right_idx = feature_column > threshold
        n, n_left, n_right = len(y), len(y[left_idx]), len(y[right_idx])
        parent_gini = self._gini(y)
        gini_left = self._gini(y[left_idx])
        gini_right = self._gini(y[right_idx])
        child_gini = (n_left / n) * gini_left + (n_right / n) * gini_right
        return parent_gini - child_gini

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
    def __init__(self, n_estimators=10, max_depth=3, min_samples_split=2, subsample=0.5, colsample=0.5):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.subsample = subsample
        self.trees = []

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.array(y)
        self.trees = []
        for i in range(self.n_estimators):
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_sample(self, X, y):
        n_samples = len(X)
        indices = np.random.choice(np.arange(n_samples), n_samples * self.subsample, replace=True)
        return X[indices], y[indices]

    def predict(self, X):
        X = np.asarray(X, float)
        tree_preds = [tree.predict(X) for tree in self.trees]
        tree_preds = list(zip(*tree_preds))
        return np.array([self._most_common_label(preds) for preds in tree_preds])

    def _most_common_label(self, labels):
        return Counter(labels).most_common(1)[0][0]