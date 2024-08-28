from abc import ABC

import numpy as np


class Activation(ABC):
    def activate(self, x: float) -> float:
        pass


class Sigmoid(Activation):
    def __init__(self) -> None:
        super().__init__()

    def activate(self, x: float) -> float:
        return 1 / (1 + np.exp(-x))


class Basic(Activation):
    def __init__(self) -> None:
        super().__init__()

    def activate(self, x: float) -> float:
        return 1 if x >= 0.0 else 0.0