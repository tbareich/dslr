from __future__ import annotations
import os
from tokenize import group
import numpy as np
import pandas as pd


class LogisticRegression:

    def __init__(self,
                 X=None,
                 Y=None,
                 n_iter=1000,
                 alpha=0.1,
                 Lambda=0,
                 mean=[],
                 std=[],
                 groups=None,
                 W=[]) -> None:
        if X is not None:
            self.X = np.insert(X, 0, 1, axis=0)
        self.Y = Y
        self.n_iter = n_iter
        self.alpha = alpha
        self.Lambda = Lambda
        self._mean = mean
        self._std = std
        self._W = W
        self._cost_array = []
        self.groups = groups

    def fit(self):
        if self.groups is not None:
            for group in self.groups:
                weights = self._gradient_decent(group)
                self._W.append(weights)
        return self

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=0)
        _h = self._h(self._W, X).T
        preds = []
        for i in range(len(_h)):
            preds.append(self.groups[np.argmax(_h[i])])
        return preds

    def save(self, path="out/weights.csv"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        out = open(path, "w")
        out.write("Mean,Std,")
        for i in range(len(self._W[0])):
            out.write(f"Weight{i}")
            if i < len(self._W[0]) - 1:
                out.write(",")
        out.write("\n")
        for i in range(len(self._mean)):
            out.write(f"{self._mean[i]},{self._std[i]},")
            if i < len(self.groups):
                for j in range(len(self._W[i])):
                    out.write(f"{self._W[i][j]}")
                    if j < len(self._W[i]) - 1:
                        out.write(",")
            out.write("\n")
        out.close()
        return self

    def score(self, Y1, Y2):
        total = 0
        mis = 0
        for i in range(len(Y1)):
            if Y1[i] == Y2[i]:
                total += 1
            else:
                mis += 1
        print(f"Missing : {mis}")
        percent = round((total / len(Y1)) * 100, 1)
        print(f"Precission: {percent}%")

    def _g(self, x):
        return 1 / (1 + np.exp(-x))

    def _h(self, theta, X):
        return self._g(np.dot(theta, X))

    def _loss(self, h, Y_ovr, m, L1):
        return (-1 / m) * np.sum(Y_ovr * np.log(h) +
                                 (1 - Y_ovr) * np.log(1 - h)) + L1

    def _gradient_decent(self, group):
        m = self.Y.shape[1]
        Y_ovr = np.where(self.Y == group, 1, 0)
        weights = np.zeros(self.X.shape[0])
        for _ in range(self.n_iter):
            _h = self._h(weights, self.X)
            L2 = (self.Lambda / m) * weights
            weights -= (self.alpha / m) * np.dot(
                (_h - Y_ovr), self.X.T)[0] + L2
        return weights