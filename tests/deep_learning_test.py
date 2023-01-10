import numpy as np
import pytest
from ml_from_scratch.MLP import MLP


class Tester:

    def test_fit(self):
        model = MLP()
        x = np.zeros(shape=(10, 150))
        y = np.zeros(shape=(10, 1))
        model.fit(x, y)
        weights, biases = model.getAllLayers()
        ans = [(150, 50),
               (50, 26),
               (26, 1)]
        for i in range(len(weights)):
            assert weights[i].shape == ans[i]

    @pytest.mark.parametrize("x_shift, y_shift",
                             [(0.1, 0.1),
                              (0.9, 0.9),
                              (0.1, 0.9),
                              (0.9, 0.1)])
    def test_feedforward(self, x_shift, y_shift):
        model = MLP()
        tolerance = 1e-4
        x = np.zeros(shape=(2, 150)) + x_shift
        y = np.zeros(shape=(2, 1)) + y_shift
        for i in range(100):
            model.fit(x, y, max_iterations=100)
            if model.cost(x, y) <= tolerance:
                break
        model.forwardPropagation(x)
        assert model.cost(x, y) <= tolerance

    def test_backprop(self):
        model = MLP()
        x = np.zeros(shape=(1, 150))
        y = np.zeros(shape=(1, 1))
        model.init_model(x.shape, y.shape)
        weights, biases = model.getAllLayers()
        d_weights, d_biases = model.backPropagation(x, y)

        for i in range(len(weights)):
            assert weights[i].shape == d_weights[i].shape
            assert biases[i].shape == d_biases[i].shape

    def test_square(self):
        model = MLP()
        assert model.square(4) == 16
