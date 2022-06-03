import numpy as np


class Statistics:

    @staticmethod
    def mean(X) -> float:
        return sum(X) / len(X)

    @staticmethod
    def std(X, mean) -> float:
        return np.sqrt(sum((X - mean)**2) / (len(X) - 1))

    @staticmethod
    def percentil(X, quartile) -> float:
        qu_p = quartile * .25 * (len(X) - 1)
        qu_i = int(qu_p)
        qu_fract = qu_p - int(qu_i)
        _percentil = X[qu_i] * (1 - qu_fract) + X[qu_i + 1] * qu_fract
        return _percentil