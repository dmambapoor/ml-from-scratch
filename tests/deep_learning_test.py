import numpy as np
from deep_learning_from_scratch.MLP import MLP


class Tester:

    def test_init(self):
        model = MLP()
        x = np.zeros(shape=(10, 150))
        y = np.zeros(shape=(10, 1))
        model.fit(x, y)
        weights, biases = model.getAllLayers()
        ans = [(150, 50),
               (50, 26),
               (26, 1)]
        print(len(weights))
        print(len(biases))
        for i in range(len(weights)):
            assert weights[i].shape == ans[i]

    def test_feedforward(self):
        model = MLP()
        x = np.zeros(shape=(10, 150))
        y = np.zeros(shape=(10, 1))
        model.fit(x, y)
        model.forwardPropagation(x)
        assert model.forwardPropagation(x)[0][0] == 0.5

    def test_backprop(self):
        model = MLP()
        x = np.zeros(shape=(1, 150))
        y = np.zeros(shape=(1, 1))
        model.fit(x, y)
        weights, biases = model.getAllLayers()
        d_weights, d_biases = model.backPropagation(x, y[0])

        for i in range(len(weights)):
            assert weights[i].shape == d_weights[i].shape
            assert biases[i].shape == d_biases[i].shape

    def test_square(self):
        model = MLP()
        assert model.square(4) == 16
