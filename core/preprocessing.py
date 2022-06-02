from core.statistics import Statistics


class Standarize:

    def __init__(self, mean=[], std=[]) -> None:
        self.mean = mean
        self.std = std

    def fit(self, X):
        self.mean = Statistics.mean(X)
        self.std = Statistics.std(X, self.mean)

    def transform(self, X):
        for i in range(0, len(X)):
            X[i] = (X[i] - self.mean[i]) / self.std[i]
        return X