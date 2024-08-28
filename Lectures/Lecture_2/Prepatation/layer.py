from random import uniform
from neuron import Neuron
from activation_functions import Activation


class Layer:
    def __init__(self, size: int, activation: Activation) -> None:
        self.size = size
        self.activation = activation
        self.neurons: list[Neuron] | None = None

    def apply(self, x: list[float]) -> list[float]:
        # This should only happen once - at the first run! Else the neurons will never update correctly!
        if self.neurons is None:
            self.neurons = [self.__create_neuron(len(x)) for _ in range(self.size)]
        return [n.feedforward(x) for n in self.neurons]

    def __create_neuron(self, weight_length: int) -> Neuron:
        initial_weights = [uniform(-2.0, 1.0) for _ in range(weight_length)]
        return Neuron(initial_weights, uniform(-2.0, 1.0), self.activation)