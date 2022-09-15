from sklearn.linear_model import LogisticRegression
from joblib import dump

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
        self.solver = LogisticRegression(
            random_state=random_state, solver='liblinear', max_iter=max_iter)

    def adjust(self):
        self.solver.fit(self.x, self.y)
        dump(self.solver, 'logistic_regression.joblib')

    def predict(self, sample):
        return self.solver.predict(sample)
