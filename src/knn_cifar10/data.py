from pathlib import Path

import numpy as np
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def load_cifar10(data_dir="data", flatten=True):
    data_dir = Path(data_dir)

    train_set = CIFAR10(root=data_dir, train=True, download=True, transform=ToTensor())
    test_set = CIFAR10(root=data_dir, train=False, download=True, transform=ToTensor())
    x_train = np.stack([img.numpy() for img, _ in train_set])
    y_train = np.array([label for _, label in train_set])

    x_test = np.stack([img.numpy() for img, _ in test_set])
    y_test = np.array([label for _, label in test_set])

    if flatten:
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)

    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_cifar10()
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
