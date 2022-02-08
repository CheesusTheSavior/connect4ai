import numpy as np


def mutate(arr, mutation_rate=0.05, mutation_factor=0.25):
    """
    mutation function that takes in an array and mutates it
    the mutation rate determines approximately how many genes are mutated.
    the mutation factor determines approximately by how much the genes are changed.
    For the changes, the standard normal distribution is used and multiplied with the mutation factor.
    """
    shape = arr.shape
    flat = arr.flatten()
    choices = np.random.random(len(flat))
    choices = [True if x < mutation_rate else False for x in choices]

    for i in range(len(choices)):
        if not choices[i]:
            continue
        else:
            flat[i] = flat[i]*(1+np.random.normal()*mutation_factor)

    arr = flat.reshape(shape)

    return arr


class GamBot:
    """
    Defines a class of neural network driven connect 4 bots.
    It is a standard forward propagating neural network.
    The default neural network is made up of an input layer with 42 inner neurons.
    Each neuron corresponds to one field of the default 6x7 Connect 4 board.
    The output consists of 7 outer neurons, each of which relates to one of the 7 playable columns.


    """

    def __init__(self, input_size=42, output_size=7, mid_layer=14, start_range=2, start_bias=1, name=None):
        if name:
            self.name = name
        self.input_size = input_size
        self.output_size = output_size
        self.mid_layer = mid_layer
        self.start_range = start_range
        self.start_bias = start_bias


        # initialize the first set of connections randomly in the starting range
        self.weight_inner = np.random.rand(mid_layer, input_size)*start_range
        # initialize the second set of connections randomly in the starting range
        self.weight_outer = np.random.rand(output_size, mid_layer)*start_range
        # initialize all biases at the start_bias in the hidden layer
        self.hidden_bias = np.random.rand(mid_layer, 1)*start_bias

    def __repr__(self):
        if hasattr(self, "name"):
            return f"GamBot {self.name}"
        else:
            return f"GamBot with no name"

    def calculate(self, input_values):
        if len(input_values) != self.input_size:
            raise ValueError(f"Input must be {self.input_size} long.")

        hidden_layer_input = self.weight_inner.dot(input_values)

        hidden_layer_activation = np.tanh(hidden_layer_input-self.hidden_bias)

        output = self.weight_outer.dot(hidden_layer_activation)

        output_activation = np.e**output/(1+np.e**output)

        return output_activation

    def full_mutate(self, mutation_rate=0.05, mutation_factor=0.25):
        self.weight_inner = mutate(self.weight_inner, mutation_rate, mutation_factor)
        self.weight_outer = mutate(self.weight_outer, mutation_rate, mutation_factor)
        self.hidden_bias = mutate(self.hidden_bias, mutation_rate, mutation_factor)

    def cross(self, parent_2):
        """
        Function that crosses two GamBots by randomly choosing genes from the one or the other parent
        :param self: Must be a GamBot
        :param parent_2: Must be a GamBot of identical size as parent 1
        :return: Returns child Gambot
        """
        if parent_2.input_size != self.input_size or \
                parent_2.output_size != self.output_size or \
                parent_2.mid_layer != self.mid_layer or \
                parent_2.start_range != self.start_range or \
                parent_2.start_bias != self.start_bias:
            raise ValueError("parents must be identical!")

        child = GamBot(self.input_size,
                       self.output_size,
                       self.mid_layer,
                       self.start_range,
                       self.start_bias)
        # take random parent genes for
        p1_w_in = self.weight_inner
        p2_w_in = self.weight_inner
        w_in_shape = p1_w_in.shape
        p1_w_in = p1_w_in.flatten()
        p2_w_in = p2_w_in.flatten()
        w_in_cross = np.random.randint(1,2, len(p1_w_in))
        c_w_in = np.array([p1_w_in[i] if i==1 else p2_w_in for i in w_in_cross]).reshape(w_in_shape)

        child.weight_inner = c_w_in

        p1_w_out = self.weight_outer
        p2_w_out = self.weight_outer
        w_out_shape = p1_w_out.shape
        p1_w_out = p1_w_out.flatten()
        p2_w_out = p2_w_out.flatten()
        w_out_cross = np.random.randint(1, 2, len(p1_w_out))
        c_w_out = np.array([p1_w_out[i] if i == 1 else p2_w_out for i in w_out_cross]).reshape(w_out_shape)

        child.weight_outer = c_w_out

        p1_b_hid = self.hidden_bias
        p2_b_hid = self.hidden_bias
        b_hid_shape = p1_b_hid.shape
        p1_b_hid = p1_b_hid.flatten()
        p2_b_hid = p2_b_hid.flatten()
        b_hid_cross = np.random.randint(1, 2, len(p1_b_hid))
        c_b_hid = np.array([p1_b_hid[i] if i == 1 else p2_b_hid for i in b_hid_cross]).reshape(b_hid_shape)

        child.hidden_bias = c_b_hid

        return child







