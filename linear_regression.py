from __future__ import annotations
import numpy as np
import csv


class LinearRegression:

    def __init__(self,
                 learning_rate=0.01,
                 theta_0=0.0,
                 theta_1=0.0,
                 epochs=0) -> None:
        self.theta_0: float = theta_0
        self.theta_1: float = theta_1
        self.learning_rate: float = learning_rate
        self.epochs: np.integer = epochs

    def fit(self) -> None:
        maxX = max(self.X)
        maxY = max(self.Y)
        self.X /= maxX
        self.Y /= maxY
        is_updated = self.__update_weights()
        if is_updated:
            self.theta_0 *= maxY
            self.theta_1 *= maxY / maxX
        self.X *= maxX
        self.Y *= maxY

    def predict(self, X) -> float:
        return self.__linear_model(X)

    def precision(self) -> float:
        Y = self.Y
        X = self.X
        m = self.m
        return sum((Y - abs(Y - self.__linear_model(X))) / Y) / m * 100

    @classmethod
    def read_csv(cls,
                 path,
                 learning_rate=0.01,
                 theta_0=0.0,
                 theta_1=0.0,
                 epochs=0) -> LinearRegression:
        try:
            X = np.array([])
            Y = np.array([])
            with open(path) as csv_file:
                csv_reader = csv.reader(csv_file)
                i = 0
                for row in csv_reader:
                    if i != 0:
                        X = np.append(X, float(row[0]))
                        Y = np.append(Y, float(row[1]))
                    i += 1
            obj = cls(learning_rate, theta_0, theta_1, epochs)
            obj.X = X
            obj.Y = Y
            obj.m = len(X)
            return obj
        except OSError:
            print('The training file does\'t exist.')
        except Exception:
            print('Something went wrong.')

    def __linear_model(self, X):
        return self.theta_0 + self.theta_1 * X

    def __update_weights(self):
        X = self.X
        Y = self.Y
        if self.epochs == 0:
            return 0
        while self.epochs:
            tmp_theta_0 = (1 / self.m) * sum(self.__linear_model(X) - Y)
            tmp_theta_1 = (1 / self.m) * sum(X * (self.__linear_model(X) - Y))
            self.theta_0 -= self.learning_rate * tmp_theta_0
            self.theta_1 -= self.learning_rate * tmp_theta_1
            self.epochs -= 1
        return 1
