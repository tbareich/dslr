# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    describe.py                                        :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: tbareich <tbareich@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2022/06/06 14:29:19 by tbareich          #+#    #+#              #
#    Updated: 2022/06/07 08:38:03 by tbareich         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from descriptive_statistic import DescriptiveStatistic
import argparse

try:
    parser = argparse.ArgumentParser(description='show data description.')
    parser.add_argument('filename',
                        type=str,
                        help='the file containing the training data.')

    args = parser.parse_args()
    filename = args.filename

    ds = DescriptiveStatistic.read_csv(filename)
    print(ds.describe())
except Exception as e:
    print(e)