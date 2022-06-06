# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    preprocessing.py                                   :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: tbareich <tbareich@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2022/06/06 14:29:31 by tbareich          #+#    #+#              #
#    Updated: 2022/06/06 14:29:32 by tbareich         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from src.statistics import Statistics


class StandarScaler:

    def __init__(self, mean=[], std=[]) -> None:
        self.mean = mean
        self.std = std

    def fit(self, X):
        self.mean = Statistics.mean(X)
        self.std = Statistics.std(X, self.mean)

    def transform(self, X):
        for i in range(0, len(X)):
            X[i] = (X[i] - self.mean[i]) / self.std[i]
        return X