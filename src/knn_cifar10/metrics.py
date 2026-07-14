import numpy as np

def accuracy_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return np.mean(y_true == y_pred)


def confusion_matrix(y_true, y_pred, num_classes=10):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[true_label, pred_label] += 1

    return matrix

def per_class_accuracy(matrix):
    matrix = np.asarray(matrix)
    return np.diag(matrix) / np.sum(matrix, axis=1)