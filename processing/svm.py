from sklearn import svm
from joblib import dump

class SupportVectorMachine:

    def __init__(self, x, y, max_iter):
        self.x = x
        self.y = y
        self.solver = svm.SVC(decision_function_shape='ovo', max_iter=max_iter)

    def fit(self):
        self.solver.fit(self.x, self.y)
        dump(self.solver, 'svm.joblib')

    def predict(self, sample):
        self.predict(sample)
