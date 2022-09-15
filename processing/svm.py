from sklearn import svm


class SupportVectorMachine:

    def __init__(self, x, y, max_iter):
        self.x = x
        self.y = y
        self.solver = svm.SVC(decision_function_shape='ovo', max_iter=max_iter)

    def fit(self):
        self.solver.fit(self.x, self.y)

    def predict(self, sample):
        self.predict(sample)
