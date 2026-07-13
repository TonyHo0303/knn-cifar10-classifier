import numpy as np


class KNNClassifier:
    def __init__(self, k=3, batch_size=100):
        if k <= 0:
            raise ValueError("k must be a positive integer.")
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")

        self.k = k
        self.batch_size = batch_size
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)

        if X_train.ndim != 2:
            raise ValueError("X_train must be a 2D array.")
        if y_train.ndim != 1:
            raise ValueError("y_train must be a 1D array.")
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("X_train and y_train must contain the same number of samples.")
        if self.k > X_train.shape[0]:
            raise ValueError("k cannot be larger than the number of training samples.")

        self.X_train = X_train
        self.y_train = y_train
        return self

    def predict(self, X_test, batch_size=None):
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model has not been fitted yet.")

        X_test = np.asarray(X_test)
        if X_test.ndim != 2:
            raise ValueError("X_test must be a 2D array.")
        if X_test.shape[1] != self.X_train.shape[1]:
            raise ValueError("X_test must have the same number of features as X_train.")

        batch_size = batch_size or self.batch_size
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")

        predictions = []

        for start in range(0, X_test.shape[0], batch_size):
            end = start + batch_size
            X_batch = X_test[start:end]

            distances = np.linalg.norm(
                self.X_train[:, np.newaxis] - X_batch,
                axis=2,
            )
            nearest_indices = np.argpartition(distances, self.k - 1, axis=0)[: self.k]

            batch_predictions = [
                np.bincount(self.y_train[nearest_indices[:, i]]).argmax()
                for i in range(X_batch.shape[0])
            ]
            predictions.extend(batch_predictions)

        return np.array(predictions)

    def score(self, X_test, y_test, batch_size=None):
        y_test = np.asarray(y_test)
        predictions = self.predict(X_test, batch_size=batch_size)
        return np.mean(predictions == y_test)
