import numpy as np
from activation_functions import Activation


class Neuron:
    def __init__(self, weights: list[float], bias: float, activation: Activation,
                 learning_rate: float = 0.01):
        self.weights = np.array(weights)
        self.bias = bias
        self.activation = activation
        self.learning_rate = learning_rate

    def feedforward(self, inputs: list[float]) -> float:
        net = self.__input_function(inputs)
        act = self.__activation_function(net, 0.0)
        ext = self.__output_function(act)
        return ext

    def back_propagate(self) -> float:
        pass

    def __input_function(self, inputs: list[float]) -> float:
        net = self.weights.dot(inputs) + self.bias
        return net

    def __activation_function(self, net: float, threshold: float) -> float:
        act = self.activation.activate(net)
        return act

    def __output_function(self, act_value: float) -> float:
        """ SOFTMAX """
        return act_value

