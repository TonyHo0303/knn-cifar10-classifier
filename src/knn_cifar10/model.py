import numpy as np


class KNNClassifier:
    def __init__(self, k=3, batch_size=100):
        self.k = k
        self.batch_size = batch_size
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test, batch_size=None):
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model has not been fitted yet.")

        batch_size = batch_size or self.batch_size
        predictions = []

        for start in range(0, X_test.shape[0], batch_size):
            end = start + batch_size
            X_batch = X_test[start:end]

            distances = np.linalg.norm(
                self.X_train[:, np.newaxis] - X_batch,
                axis=2,
            )
            nearest_indices = np.argsort(distances, axis=0)[: self.k]

            batch_predictions = [
                np.bincount(self.y_train[nearest_indices[:, i]]).argmax()
                for i in range(X_batch.shape[0])
            ]
            predictions.extend(batch_predictions)

        return np.array(predictions)
