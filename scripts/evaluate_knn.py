from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from knn_cifar10.data import load_cifar10
from knn_cifar10.model import KNNClassifier


def main():
    (X_train, y_train), (X_test, y_test) = load_cifar10(flatten=True)

    X_train = X_train[:2000]
    y_train = y_train[:2000]
    X_test = X_test[:200]
    y_test = y_test[:200]

    for i in 1, 3, 5, 7:
        model = KNNClassifier(k=i, batch_size=10)
        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)
        print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
