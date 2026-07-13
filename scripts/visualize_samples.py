from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from knn_cifar10.data import CIFAR10_CLASSES, load_cifar10


def visualize_samples(num_samples=16, seed=42):
    (x_train, y_train), _ = load_cifar10(flatten=False)

    rng = np.random.default_rng(seed)
    sample_indices = rng.choice(len(x_train), size=num_samples, replace=False)

    cols = 4
    rows = int(np.ceil(num_samples / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.4))
    axes = np.asarray(axes).reshape(-1)

    for ax, sample_index in zip(axes, sample_indices):
        image = x_train[sample_index].transpose(1, 2, 0)
        label = y_train[sample_index]

        ax.imshow(image)
        ax.set_title(CIFAR10_CLASSES[label])
        ax.axis("off")

    for ax in axes[num_samples:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_samples()
