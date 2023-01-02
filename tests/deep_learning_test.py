from deep_learning_from_scratch.MLP import MLP


class Tester:

    def test_square(self):
        model = MLP()
        assert model.square(4) == 16
