from deep_learning_from_scratch.DNN import DNN


class Tester:

    def test_square(self):
        model = DNN()
        assert model.square(4) == 16
