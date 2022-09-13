import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier


class LogisticRegressionModel:
    def __init__(self, x, y, fit_intercept, intercept_scaling, random_state, max_iter):
        self.x = x
        self.y = y
        self.lr = LogisticRegression(
            fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, random_state=random_state, solve='liblinear', max_iter=max_iter)

    def adjust(self):
        self.lr.fit(self.x, self.y)

    def predict(self, sample):
        return self.lr.predict(sample)
