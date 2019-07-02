import numpy as np


class AdalineGD:
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        """

        """
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1+X.shape[1])

        self.cost_ = []

        for _ in range(self.n_iter):
            cost = 0
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0

            self.cost_.append(cost)

        return self

    def net_input(self, X):
        """
        calculate net_input by linear combination of weights and feature matrix X
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, net_input):
        """Compute linear activation"""
        return net_input

    def predict(self, sample_matrix):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(sample_matrix)) >= 0.0, 1, -1)
