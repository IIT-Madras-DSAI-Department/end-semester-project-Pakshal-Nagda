import numpy as np
import pandas as pd

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
