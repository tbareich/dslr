import argparse
import os
import pandas as pd

from src.logistic_regression import LogisticRegression
from src.preprocessing import StandarScaler

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

    standarize = StandarScaler(mean=mean, std=std)
    X = standarize.transform(X.T)
    predictions = LogisticRegression(groups=groups, weights=weights).predict(X)

    filename = "out/house.csv"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    out = open(filename, "w")
    df = pd.DataFrame()
    df["Index"] = list(range(len(predictions)))
    df["Hogwarts House"] = predictions
    df.to_csv(filename, index=False)
except Exception as e:
    print(e)