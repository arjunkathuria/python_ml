import numpy as np


class Perceptron:

    """
    parameters:-
    ----------

    eta: float
         learning rate (number b/w 0.0 and 1.0)

    n_iter: int
         iterations over training set.

    random_state: int
         Random number generator seed.
    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, Y):
        """Fits the training data onto the model.

        parameters:-
        ----------

        X : {feature matrix with training samples, array like (numpy object)},
            shape = [n_samples, n_features]

        y : {output vector with target values, array like (numpy object)}
            shape = [n_samples]
        """

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1+X.shape[1])

        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, Y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculates the net input"""

        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Returns the class label after applying unit step"""

        return np.where(self.net_input(X) >= 0.0, 1, -1)
