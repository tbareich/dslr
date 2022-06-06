# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    score.py                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: tbareich <tbareich@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2022/06/06 14:29:08 by tbareich          #+#    #+#              #
#    Updated: 2022/06/06 14:29:09 by tbareich         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from sklearn.metrics import accuracy_score

import pandas as pd

house = pd.read_csv("out/house.csv")
truth = pd.read_csv("datasets/dataset_truth.csv")

print(
    accuracy_score(truth["Hogwarts House"].values,
                   house["Hogwarts House"].values))
