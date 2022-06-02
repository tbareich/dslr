import argparse
import os
import re
import pandas as pd

from core.logistic_regression import LogisticRegression
from core.preprocessing import Standarize

try:
    parser = argparse.ArgumentParser(description='run your model')
    parser.add_argument('filename',
                        type=str,
                        help='file containing the test data')
    parser.add_argument("-w",
                        '--weights',
                        type=str,
                        help='file containing the weights',
                        required=True)
    args = parser.parse_args()

    features = [
        "Defense Against the Dark Arts", "Herbology", "Divination", "Charms",
        "Ancient Runes", "Muggle Studies", "History of Magic",
        "Transfiguration"
    ]

    groups = ["Hufflepuff", "Ravenclaw", "Gryffindor", "Slytherin"]

    pred_file = pd.read_csv(args.filename)
    pred_file = pred_file.ffill()
    X = pred_file[features].values

    weights_file = pd.read_csv(args.weights)
    weights = weights_file.values[:4, 2:11]
    mean = weights_file.values[:8, :1].T[0]
    std = weights_file.values[:8, 1:2].T[0]

    standarize = Standarize(mean=mean, std=std)
    X = standarize.transform(X.T)
    predictions = LogisticRegression(Lambda=0,
                                     groups=groups,
                                     mean=standarize.mean,
                                     std=standarize.std,
                                     W=weights).predict(X)

    filename = "out/house.csv"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    out = open(filename, "w")

    out.write("Index,Hogwarts House\n")
    for i in range(len(predictions)):
        out.write(f"{i},{predictions[i]}")
        if i < len(predictions) - 1:
            out.write("\n")
    out.close()
except Exception as e:
    print(e)