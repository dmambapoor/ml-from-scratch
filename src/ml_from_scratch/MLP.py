import numpy as np
from scipy import special as sp  # type: ignore


class MLP:

    def __init__(self):
        pass

    def init_model(self, x_shape, y_shape, learning_rate=1, regularization=1, batch_size=None, max_iterations=100, tolerance=1e-4, random_state=None, verbose=False):
        # Determine the sizes of the layers and biases -> add them to the array of layers
        self.n_inputs = x_shape[1]
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.batch_size = int(x_shape[0]//10) + 1 if (not batch_size) or batch_size > x_shape[0] else batch_size
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.random_state = random_state
        self.verbose = verbose
        self.num_layers = 4
        self.weights = []
        self.biases = []

        if random_state:
            np.random.seed(random_state)

        w1 = (np.random.rand(self.n_inputs, (self.n_inputs//2 + 1 if self.n_inputs//2 < 50 else 50)) - 0.5) * 10
        self.weights.append(w1)
        b1 = (np.random.rand(1, w1.shape[1]) - 0.5) * 10
        self.biases.append(b1)

        w2 = (np.random.rand(w1.shape[1], w1.shape[1]//2 + 1) - 0.5)*10
        self.weights.append(w2)
        b2 = (np.random.rand(1, w2.shape[1]) - 0.5)*10
        self.biases.append(b2)

        w3 = (np.random.rand(w2.shape[1], y_shape[1]) - 0.5)*10
        self.weights.append(w3)
        b3 = (np.random.rand(w3.shape[1], y_shape[1]) - 0.5)*10
        self.biases.append(b3)

    # Construct and train the multi-layer perceptrons
    def fit(self, x: np.ndarray, y: np.ndarray, learning_rate=1, regularization=1, batch_size=None, random_state=None, max_iterations=100, tolerance=1e-4, verbose=False):
        # Add two hidden layers and one final layer consisting of only one node.
        self.init_model(x.shape, y.shape, learning_rate=learning_rate, regularization=regularization, batch_size=batch_size, max_iterations=max_iterations, random_state=random_state, verbose=verbose)

        # ----MAIN_LEARNING_LOOP-----

        # Initialize vars
        c_cost = self.cost(x, y)
        p_cost = c_cost
        failed_iterations = 0
        failed_iterations_threshold = 1000
        failed_iterations_punishment = 2
        iteration_num = 0

        # Loop gradient descent
        while c_cost > self.tolerance and iteration_num < self.max_iterations:
            # Take a gradient descent step if we're slowing down
            if p_cost - c_cost < (tolerance / self.max_iterations):
                self.gradient_descent_step(x, y)
            else:
                self.stochastic_gradient_descent_step(x, y)

            # Update failure tracking
            p_cost = c_cost
            c_cost = self.cost(x, y)
            if verbose:
                print("ITERATION #%i COST: %f (change of %f)" % (iteration_num, c_cost, c_cost-p_cost))
            if c_cost >= p_cost:
                failed_iterations += 1
            if failed_iterations > failed_iterations_threshold:
                self.learning_rate /= failed_iterations_punishment
                p_cost = c_cost
                failed_iterations = 0
                if verbose:
                    print("ITERATION #%i: Failed to improve. Reducing learning rate." % (iteration_num))

            # Update iteration number
            iteration_num += 1
        if verbose:
            if c_cost <= tolerance:
                print("Cost lower than tolerance. Stopping training.")
            elif iteration_num >= max_iterations:
                print("Max_iterations achieved. Stopping training.")

    def stochastic_gradient_descent_step(self, x, y):
        sum_d_weights = [np.zeros(shape=i.shape) for i in self.weights]
        sum_d_biases = [np.zeros(shape=i.shape) for i in self.biases]

        # Sum all the derivatives of the weights and biases
        rng = np.random.default_rng(seed=self.random_state)
        for i in range(self.batch_size):
            select_index = int(rng.random() * len(x))
            d_weights, d_biases = self.backPropagation(np.atleast_2d(x[select_index]), np.atleast_2d(y[select_index]))
            for i in range(len(sum_d_weights)):
                sum_d_weights[i] += d_weights[i]
                sum_d_biases[i] += d_biases[i]

        # Subtract the average of the derivatives above from the weights.
        # This is gradient descent.
        for i in range(len(sum_d_weights)):
            self.weights[i] -= (sum_d_weights[i] / self.batch_size) * self.learning_rate
            self.biases[i] -= (sum_d_biases[i] / self.batch_size) * self.learning_rate
    def gradient_descent_step(self, x, y):
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
            self.weights[i] -= (sum_d_weights[i] / x.shape[0]) * self.learning_rate
            self.biases[i] -= (sum_d_biases[i] / x.shape[0]) * self.learning_rate

    def cost(self, x, y):
        total_cost = 0
        for i in range(x.shape[0]):
            total_cost += (self.forwardPropagation(np.atleast_2d(x[i])) - np.atleast_2d(y[i])) ** 2
        total_cost /= x.shape[0]
        return total_cost

    def forwardPropagation(self, x):
        # Given a set of inputs, calculate the output of each layer and feed as inputs to the next layer
        for i in range(len(self.weights)):
            x = sp.expit(np.add(x @ self.weights[i], self.biases[i]))  # apply sigmoid activation

        # Return the output of the last-layer
        return x

    def backPropagation(self, x: np.ndarray, y: np.ndarray):
        # Forward propogation to store activations, and linear weight combinations
        activations = [x]
        zs = []
        d_weights = [np.zeros(shape=i.shape) for i in self.weights]
        d_biases = [np.zeros(shape=i.shape) for i in self.biases]

        # Define derivative of activation function : currently hard-coded as the sigmoid derivative function
        def d_sigmoid(el):
            return sp.expit(el) * (1 - sp.expit(el))

        # Given a set of inputs, calculate the output of each layer and feed as inputs to the next layer
        for i in range(len(self.weights)):
            zs.append(np.add(np.matmul(activations[i], self.weights[i]), self.biases[i]))
            activations.append(sp.expit(zs[i]))  # apply sigmoid activation function

        # Define derivative of loss function : currently hard-coded as sigmoid function
        def d_cost(y, pred_y):
            return 2 * (pred_y - y)

        # Set the derivatives of the last set of weights
        d_biases[-1] = np.matmul(d_cost(y, activations[-1]), d_sigmoid(zs[-1]))
        d_weights[-1] = np.matmul(activations[-2].transpose(), d_biases[-1])

        for i in range(2, self.num_layers):
            d_biases[-i] = np.matmul(d_biases[-i + 1], self.weights[-i + 1].transpose()) * d_sigmoid(zs[-i])
            d_weights[-i] = np.matmul(activations[-i - 1].transpose(), d_biases[-i])
        return d_weights, d_biases

    def getAllLayers(self):
        return self.weights, self.biases

    def square(self, x):
        return x * x
