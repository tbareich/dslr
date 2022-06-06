# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    logreg_predict.py                                  :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: tbareich <tbareich@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2022/06/06 14:28:28 by tbareich          #+#    #+#              #
#    Updated: 2022/06/06 14:28:30 by tbareich         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import argparse
import os
import numpy as np
import pandas as pd

from src.logistic_regression import LogisticRegression
from src.preprocessing import StandarScaler

try:
    parser = argparse.ArgumentParser(description='run your model')
    parser.add_argument('pred_filename',
                        type=str,
                        help='file containing the test data')
    parser.add_argument('weights_filename',
                        type=str,
                        help='file containing the weights')
    args = parser.parse_args()

    features = [
        "Herbology",
        "Defense Against the Dark Arts",
        "Divination",
        "Muggle Studies",
        "Ancient Runes",
        "History of Magic",
        "Transfiguration",
        "Charms",
    ]

    groups = ["Hufflepuff", "Ravenclaw", "Gryffindor", "Slytherin"]

    pred_file = pd.read_csv(args.pred_filename)
    pred_file = pred_file.ffill()
    X = pred_file[features].values

    weights_file = pd.read_csv(args.weights_filename)

    weights = weights_file.values[:4, 2:11]
    mean = weights_file.values[:8, :1].T[0]
    std = weights_file.values[:8, 1:2].T[0]

    standarize = StandarScaler(mean=mean, std=std)
    X = standarize.transform(X.T)
    predictions = LogisticRegression(groups=groups, weights=weights).predict(X)

    filename = "out/house.csv"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df = pd.DataFrame()
    df["Index"] = list(range(len(predictions)))
    df["Hogwarts House"] = predictions
    df.to_csv(filename, index=False)
except Exception as e:
    print(e)