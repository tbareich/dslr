from __future__ import annotations
import os
from tokenize import group
import numpy as np
import pandas as pd


class LogisticRegression:

    def __init__(self,
                 n_iter=1000,
                 alpha=0.1,
                 Lambda=0,
                 mean=[],
                 std=[],
                 groups=None,
                 algo="bgd",
                 batch_size=300,
                 weights=np.array([])) -> None:

        self.n_iter = n_iter
        self.alpha = alpha
        self.Lambda = Lambda
        self._mean = mean
        self._std = std
        self._W = weights
        self.algo = algo
        self.batch_size = batch_size
        self._cost_history = {}
        self.groups = groups

    def fit(self, X=None, y=None):
        if X is not None and y is not None:
            X = np.insert(X, 0, 1, axis=0)
            if self.groups is not None:
                weights_list = []
                for group in self.groups:
                    self._cost_history[group] = np.array([])
                    if self.algo == 'sgd':
                        weights = self._stochastic_gradient_descent(
                            X, y, group)
                    elif self.algo == 'mbgd':
                        weights = self._mini_batch_gradient_descent(
                            X, y, group)
                    elif self.algo == 'bgd':
                        weights = self._gradient_descent(X, y, group)
                    else:
                        raise Exception("unkonwn optimization algorithm")
                    weights_list.append(weights)
                self._W = np.array(weights_list)
        return self

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=0)
        y_pred = self._h(self._W, X).T
        preds = []
        for i in range(len(y_pred)):
            # print(y_pred[i])
            preds.append(self.groups[np.argmax(y_pred[i])])
        return preds

    def get_cost_history(self):
        return self._cost_history

    def save(self, path="out/weights.csv"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df = pd.DataFrame()
        df["Mean"] = self._mean
        df["Std"] = self._std
        weights = self._W.T
        for i in range(len(weights)):
            w_df = pd.DataFrame({f"Weight{i}": weights[i]})
            df = pd.concat([df, w_df], axis=1)
        df.to_csv(path, index=False)
        return self

    def _g(self, X):
        return 1 / (1 + np.exp(-X))

    def _f(self, theta, X):
        return np.dot(theta, X)

    def _h(self, theta, X):
        return self._g(self._f(theta, X))

    def _loss(self, pred_y, y_ovr, m, l2):
        return (-1 / m) * np.sum(y_ovr * np.log(pred_y) +
                                 (1 - y_ovr) * np.log(1 - pred_y)) + np.sum(l2)

    def _gradient_descent(self, X, y, group):
        m = y.shape[1]
        y_ovr = np.where(y == group, 1, 0)
        weights = np.zeros(X.shape[0])
        for _ in range(self.n_iter):
            pred_y = self._h(weights, X)
            l2 = (self.Lambda / 2) * weights**2
            l = self._loss(pred_y, y_ovr, m, l2)
            self._cost_history[group] = np.append(self._cost_history[group], l)

            dl2 = self.Lambda * weights
            weights -= (self.alpha / m) * np.dot(
                (pred_y - y_ovr), X.T)[0] + dl2
        return weights

    def _stochastic_gradient_descent(self, X, y, group):
        m = y.shape[1]
        y_ovr = np.where(y == group, 1, 0)
        weights = np.zeros(X.shape[0])
        for _ in range(self.n_iter):
            column_index = np.random.choice(X.shape[1], replace=False)
            x = X[:, column_index].reshape((X.shape[0], 1))
            pred_y = self._h(weights, x)
            l2 = (self.Lambda / 2) * weights**2
            l = self._loss(pred_y, y_ovr[0][column_index], 1, l2)
            self._cost_history[group] = np.append(self._cost_history[group], l)

            dl2 = self.Lambda * weights
            weights -= self.alpha * np.dot(
                (pred_y - y_ovr[0][column_index]), x.T)
        return weights

    def _mini_batch_gradient_descent(self, X, y, group):
        m = y.shape[1]
        y_ovr = np.where(y == group, 1, 0)
        weights = np.zeros(X.shape[0])
        for _ in range(self.n_iter):
            columns = np.random.choice(X.shape[1],
                                       self.batch_size,
                                       replace=False)
            new_X = X[:, columns]
            pred_y = self._h(weights, new_X)
            l2 = (self.Lambda / 2) * weights**2
            l = self._loss(pred_y, y_ovr[0][columns], self.batch_size, l2)
            self._cost_history[group] = np.append(self._cost_history[group], l)

            dl2 = self.Lambda * weights
            weights -= (self.alpha / self.batch_size) * np.dot(
                (pred_y - y_ovr[0][columns]), new_X.T) + dl2
        return weights