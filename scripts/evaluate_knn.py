from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from knn_cifar10.data import load_cifar10
from knn_cifar10.model import KNNClassifier
from knn_cifar10.metrics import accuracy_score, confusion_matrix, per_class_accuracy


def main():
    (X_train, y_train), (X_test, y_test) = load_cifar10(flatten=True)

    X_train = X_train[:2000]
    y_train = y_train[:2000]
    X_test = X_test[:200]
    y_test = y_test[:200]

    for i in 1, 3, 5, 7:
        model = KNNClassifier(k=i, batch_size=10)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        matrix = confusion_matrix(y_test, predictions)
        per_class_acc = per_class_accuracy(matrix)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Confusion Matrix:\n{matrix}")
        print(f"Per-Class Accuracy: {per_class_acc}")


if __name__ == "__main__":
    main()
