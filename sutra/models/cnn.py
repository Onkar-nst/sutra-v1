"""A compact convolutional network for 28x28 digit classification."""
from __future__ import annotations

from ..nn import BatchNorm2d, Conv2d, Dropout, Flatten, Linear, MaxPool2d, Module, ReLU, Sequential


class DigitCNN(Module):
    """conv(1→16)-BN-ReLU-pool → conv(16→32)-BN-ReLU-pool → FC(1568→128)-ReLU-dropout → FC(128→10)"""

    def __init__(self, c1: int = 16, c2: int = 32, hidden: int = 128, dropout: float = 0.25):
        super().__init__()
        self.features = Sequential(
            Conv2d(1, c1, 3, padding=1), BatchNorm2d(c1), ReLU(), MaxPool2d(2),
            Conv2d(c1, c2, 3, padding=1), BatchNorm2d(c2), ReLU(), MaxPool2d(2),
        )
        self.classifier = Sequential(
            Flatten(), Linear(c2 * 7 * 7, hidden), ReLU(), Dropout(dropout), Linear(hidden, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class MLP(Module):
    """Plain multilayer perceptron, optionally with BatchNorm — used by the ablation studies."""

    def __init__(self, sizes, batchnorm: bool = False, activation=ReLU):
        super().__init__()
        from ..nn import BatchNorm1d
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                if batchnorm:
                    layers.append(BatchNorm1d(sizes[i + 1]))
                layers.append(activation())
        self.net = Sequential(*layers)

    def forward(self, x):
        return self.net(x.flatten(1))
