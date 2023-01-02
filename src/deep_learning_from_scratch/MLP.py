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

    def feedforward(self, x):
        pass

    def getAllLayers(self):
        return self.weights, self.biases

    def square(self, x):
        return x * x
