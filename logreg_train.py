import argparse
import os
import pandas as pd
import numpy as np

from core.logistic_regression import LogisticRegression
from core.preprocessing import Standarize

try:
    parser = argparse.ArgumentParser(description='train your model')
    parser.add_argument('filename',
                        type=str,
                        help='file containing the training data')
    parser.add_argument('-i',
                        '--n_iter',
                        type=int,
                        default=1000,
                        help='number of iterations')
    parser.add_argument('-a',
                        '--alpha',
                        type=float,
                        default=0.1,
                        help='learning rate')
    parser.add_argument('-l',
                        '--Lambda',
                        type=float,
                        default=0.1,
                        help='learning rate')
    parser.add_argument("-o",
                        '--out',
                        type=str,
                        default=0.1,
                        help='weights output file')

    args = parser.parse_args()

    features = [
        "Defense Against the Dark Arts", "Herbology", "Divination", "Charms",
        "Ancient Runes", "Muggle Studies", "History of Magic",
        "Transfiguration"
    ]
    groups = ["Hufflepuff", "Ravenclaw", "Gryffindor", "Slytherin"]

    data = pd.read_csv(args.filename)
    data = data.dropna(subset=["Defense Against the Dark Arts"])
    data = data.dropna(subset=["Herbology"])
    data = data.dropna(subset=["Divination"])
    data = data.dropna(subset=["Charms"])
    data = data.dropna(subset=["Ancient Runes"])
    data = data.dropna(subset=["Muggle Studies"])
    data = data.dropna(subset=["History of Magic"])
    data = data.dropna(subset=["Transfiguration"])

    Y = data["Hogwarts House"].values
    Y = Y.reshape((Y.shape[0], 1)).T
    X = data[features].values
    standarize = Standarize()
    standarize.fit(X)
    X = standarize.transform(X.T)
    LogisticRegression(X,
                       Y,
                       Lambda=args.Lambda,
                       groups=groups,
                       mean=standarize.mean,
                       n_iter=args.n_iter,
                       alpha=args.alpha,
                       std=standarize.std).fit().save(path=args.out)

except Exception as e:
    print(e)