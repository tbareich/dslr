# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    logreg_train.py                                    :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: tbareich <tbareich@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2022/06/06 14:28:50 by tbareich          #+#    #+#              #
#    Updated: 2022/06/06 14:28:51 by tbareich         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from src.logistic_regression import LogisticRegression
from src.preprocessing import StandarScaler

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
                        default=0,
                        help='learning rate')
    parser.add_argument("-o",
                        '--out',
                        type=str,
                        default="out/weights.csv",
                        help='weights output file')
    parser.add_argument('--algo',
                        type=str,
                        default="bgd",
                        help='choose optimization algorithm')
    parser.add_argument('--batch_size',
                        type=int,
                        default=300,
                        help='choose optimization algorithm')
    parser.add_argument('--show_cost_hist',
                        action='store_true',
                        help='show cost history')

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

    data = pd.read_csv(args.filename)
    data = data.dropna(subset=["Defense Against the Dark Arts"])
    data = data.dropna(subset=["Herbology"])
    data = data.dropna(subset=["Divination"])
    data = data.dropna(subset=["Charms"])
    data = data.dropna(subset=["Ancient Runes"])
    data = data.dropna(subset=["Muggle Studies"])
    data = data.dropna(subset=["History of Magic"])
    data = data.dropna(subset=["Transfiguration"])

    y = data["Hogwarts House"].values
    y = y.reshape((1, y.shape[0]))
    X = data[features].values
    standarize = StandarScaler()
    standarize.fit(X)
    X = standarize.transform(X.T)
    lr = LogisticRegression(
        n_iter=args.n_iter,
        alpha=args.alpha,
        Lambda=args.Lambda,
        mean=standarize.mean,
        std=standarize.std,
        groups=groups,
        algo=args.algo,
        batch_size=args.batch_size,
    ).fit(X=X, y=y).save(path=args.out)

    if args.show_cost_hist:
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        plt.title("cost vs ephoc")
        history = lr.get_cost_history()
        i = 0
        for group in groups:
            cost = history[group]
            y = i % 2
            x = int(i / 2)
            axs[y, x].plot(list(range(args.n_iter)), cost)
            axs[y, x].set_xlabel("Ephochs")
            axs[y, x].set_ylabel("Cost")
            axs[y, x].set_title(group, fontsize=16)
            i += 1
        fig.suptitle('Cost vs Ephocs', fontsize=20)
        plt.show()

except Exception as e:
    print(e)