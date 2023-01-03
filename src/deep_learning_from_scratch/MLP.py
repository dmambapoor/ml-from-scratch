import numpy as np


class MLP:

    def __init__(self):
        pass

    def fit(self, x: np.ndarray, y: np.ndarray):
        # Construct multi-layer perceptron

        # Add two hidden layers and one final layer consisting of only one node.

        # Determine the sizes of the layers and biases -> add them to the array of layers
        self.n_inputs = x.shape[1]
        self.weights = []
        self.biases = []

        w1 = np.zeros(shape=(self.n_inputs, (self.n_inputs//2 + 1 if self.n_inputs//2 < 50 else 50)))
        self.weights.append(w1)
        b1 = np.zeros(shape=(1, w1.shape[1]))
        self.biases.append(b1)

        w2 = np.zeros(shape=(w1.shape[1], w1.shape[1]//2 + 1))
        self.weights.append(w2)
        b2 = np.zeros(shape=(1, w2.shape[1]))
        self.biases.append(b2)

        w3 = np.zeros(shape=(w2.shape[1], 1))
        self.weights.append(w3)
        b3 = np.zeros(shape=(w3.shape[1], 1))
        self.biases.append(b3)

        self.num_layers = 4

    def forwardPropagation(self, x):
        # Define activation function : currently hard-coded as sigmoid function
        def sigmoid(el):
            return 1.0 / (1.0 + np.exp(el))

        # Given a set of inputs, calculate the output of each layer and feed as inputs to the next layer
        for i in range(len(self.weights)):
            x = np.apply_along_axis(sigmoid, 1, x @ self.weights[i] + self.biases[i])

        # Return the output of the last-layer
        return x

    def backPropagation(self, x: np.ndarray, y: np.ndarray):
        # Forward propogation to store activations, and linear weight combinations
        print(x.shape)
        activations = [x]
        zs = []
        d_weights = [np.zeros(shape=i.shape) for i in self.weights]
        d_biases = [np.zeros(shape=i.shape) for i in self.biases]

        # Define activation function : currently hard-coded as sigmoid function
        def sigmoid(el):
            return 1.0 / (1.0 + np.exp(el))

        def d_sigmoid(el):
            return sigmoid(el) * (1 - sigmoid(el))

        # Given a set of inputs, calculate the output of each layer and feed as inputs to the next layer
        for i in range(len(self.weights)):
            zs.append(activations[i] @ self.weights[i] + self.biases[i])
            activations.append(np.apply_along_axis(sigmoid, 1, zs[i]))

        # Define loss function : currently hard-coded as sigmoid function
        def cost(y, pred_y):
            return (pred_y - y) ** 2

        def d_cost(y, pred_y):
            return 2 * (pred_y - y)

        # Set the derivatives of the last set of weights
        d_biases[-1] = d_cost(y, activations[-1]) @ d_sigmoid(zs[-1])
        d_weights[-1] = activations[-2].transpose() @ d_biases[-1]
        print(d_cost(y, activations[-1]).shape, d_sigmoid(zs[-2]).shape, d_weights[-1].shape, d_biases[-1].shape, self.weights[-1].shape)
        for i in range(2, self.num_layers):
            d_biases[-i] = (d_biases[-i + 1] @ self.weights[-i + 1].transpose()) * d_sigmoid(zs[-i])
            d_weights[-i] = activations[-i - 1].transpose() @ d_biases[-i]
        return d_weights, d_biases

    def getAllLayers(self):
        return self.weights, self.biases

    def square(self, x):
        return x * x
