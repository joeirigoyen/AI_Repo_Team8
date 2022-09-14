import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier


class LogisticRegressionModel:
    """
    Params:

    random_state[int] = Used when solver == 'sag', 'saga' or 'liblinear' to shuffle the data. 
                        RandomState instance, default=None
    max_iter[int] = Maximum number of iterations taken for the solvers to converge. default=100
    """

    def __init__(self, x, y, random_state, max_iter):
        self.x = x
        self.y = y
        self.regressor = LogisticRegression(
            random_state=random_state, solve='liblinear', max_iter=max_iter)

    def adjust(self):
        self.regressor.fit(self.x, self.y)

    def predict(self, sample):
        return self.regressor.predict(sample)
