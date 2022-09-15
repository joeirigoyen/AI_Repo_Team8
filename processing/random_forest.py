from sklearn.ensemble import RandomForestClassifier
from joblib import dump


class RFClassifier:
    def __init__(self, x, y, trees, quality_criterion, depth, min_split, min_leaf, features):
        self.x = x
        self.y = y
        self.solver = RandomForestClassifier(n_estimators=trees, criterion=quality_criterion, max_depth=depth, min_samples_split=min_split, min_samples_leaf=min_leaf, max_features=features, n_jobs=-1)

    def adjust(self):
        self.solver.fit(self.x, self.y)
        dump(self.solver, 'random_forest.joblib')

    def predict(self, sample):
        return self.solver.predict(sample)
