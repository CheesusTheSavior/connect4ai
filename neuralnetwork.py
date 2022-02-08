import random
import numpy as np


class GamBot:
    """
    Defines a class of neural network driven connect 4 bots.
    It is a standard forward propagating neural network.
    The default neural network is made up of an input layer with 42 inner neurons.
    Each neuron corresponds to one field of the default 6x7 Connect 4 board.
    The output consists of 7 outer neurons, each of which relates to one of the 7 playable columns.


    """

    def __init__(self, input_size=42, output_size=7, mid_layer=14, start_range=2, start_bias=1):
        self.name = '%030x' % random.randrange(16**30)
        self.input_size = input_size
        # initialize the first set of connections randomly in the starting range
        self.weight_inner = np.random.rand(mid_layer, input_size)*start_range
        # initialize the second set of connections randomly in the starting range
        self.weight_outer = np.random.rand(output_size, mid_layer)*start_range
        # initialize all biases at the start_bias in the hidden layer
        self.hidden_bias = np.ones((mid_layer, 1))*start_bias

    def __repr__(self):

        return f"Hello, it's me, {self.name}! I am a GamBot!"

    def calculate(self, input_values):
        if len(input_values) != self.input_size:
            raise ValueError(f"Input must be {self.input_size} long.")

        hidden_layer_input = self.weight_inner.dot(input_values)

        hidden_layer_activation = np.tanh(hidden_layer_input-self.hidden_bias)

        output = self.weight_outer.dot(hidden_layer_activation)

        output_activation = np.e**output/(1+np.e**output)

        return output_activation



