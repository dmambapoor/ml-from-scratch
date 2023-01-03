import numpy as np


class MLP:

    def __init__(self):
        pass

    def init_model(self, x_shape, y_shape, learning_rate=1, regularization=1, max_iterations=100):
        # Determine the sizes of the layers and biases -> add them to the array of layers
        self.n_inputs = x_shape[1]
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.max_iterations = max_iterations
        self.num_layers = 4
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

        w3 = np.zeros(shape=(w2.shape[1], y_shape[1]))
        self.weights.append(w3)
        b3 = np.zeros(shape=(w3.shape[1], y_shape[1]))
        self.biases.append(b3)

    # Construct and train the multi-layer perceptrons
    def fit(self, x: np.ndarray, y: np.ndarray, learning_rate=1, regularization=1, max_iterations=100, tolerance=1e-5):
        # Add two hidden layers and one final layer consisting of only one node.
        self.init_model(x.shape, y.shape, learning_rate, regularization)

        # ----MAIN_LEARNING_LOOP-----
        c_cost = self.cost(x, y)
        p_cost = c_cost
        groove_cost = c_cost
        failed_iterations = 0
        total_failed_iterations = 0
        while c_cost > tolerance:
            sum_d_weights = [np.zeros(shape=i.shape) for i in self.weights]
            sum_d_biases = [np.zeros(shape=i.shape) for i in self.biases]
            # Sum all the derivatives of the weights and biases
            for i in range(len(x)):
                d_weights, d_biases = self.backPropagation(np.atleast_2d(x[i]), np.atleast_2d(y[i]))
                for i in range(len(sum_d_weights)):
                    sum_d_weights[i] += d_weights[i]
                    sum_d_biases[i] += d_biases[i]

            # Subtract the average of the derivatives above from the weights.
            # This is gradient descent.
            for i in range(len(sum_d_weights)):
                self.weights[i] += (sum_d_weights[i] / x.shape[0]) * self.learning_rate
                self.biases[i] += (sum_d_biases[i] / x.shape[0]) * self.learning_rate
            
            p_cost = c_cost
            c_cost = self.cost(x, y)
            if c_cost > p_cost:
                failed_iterations += 1
                total_failed_iterations += 1
            if failed_iterations > 10:
                learning_rate /= 10
                p_cost = c_cost
                groove_cost = c_cost
                failed_iterations = 0
            if failed_iterations > 1000:
                break
            if c_cost < (groove_cost // 10):
                learning_rate /= 10
                p_cost = c_cost
                groove_cost = c_cost
            
    def cost(self, x, y):
        total_cost = 0
        for i in range(x.shape[0]):
            total_cost += (self.forwardPropagation(np.atleast_2d(x[i])) - y[i]) ** 2
        total_cost /= x.shape[0]
        return total_cost

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

        # Define derivative of loss function : currently hard-coded as sigmoid function
        def d_cost(y, pred_y):
            return 2 * (pred_y - y)

        # Set the derivatives of the last set of weights
        d_biases[-1] = d_cost(y, activations[-1]) @ d_sigmoid(zs[-1])
        d_weights[-1] = activations[-2].transpose() @ d_biases[-1]

        for i in range(2, self.num_layers):
            d_biases[-i] = (d_biases[-i + 1] @ self.weights[-i + 1].transpose()) * d_sigmoid(zs[-i])
            d_weights[-i] = activations[-i - 1].transpose() @ d_biases[-i]
        return d_weights, d_biases

    def getAllLayers(self):
        return self.weights, self.biases

    def square(self, x):
        return x * x
