from deep_learning_test import Tester

model = Tester()
model.test_fit()
model.test_feedforward(0.1, 0.1)
model.test_feedforward(0.1, 0.9)
model.test_feedforward(0.9, 0.1)
model.test_feedforward(0.9, 0.9)
