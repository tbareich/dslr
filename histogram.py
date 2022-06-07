# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    histogram.py                                       :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: tbareich <tbareich@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2022/06/06 14:29:14 by tbareich          #+#    #+#              #
#    Updated: 2022/06/07 08:51:54 by tbareich         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from src.descriptive_statistic import DescriptiveStatistic
import argparse

try:
    parser = argparse.ArgumentParser(description='show data histogram plot.')

    parser.add_argument('filename',
                        type=str,
                        help='the file containing the training data.')
    parser.add_argument('--show_all',
                        action='store_const',
                        const=True,
                        default=False,
                        help='show all features histogram.')

    args = parser.parse_args()
    filename = args.filename
    show_all = args.show_all

    ds = DescriptiveStatistic.read_csv(filename)

    ds.show_histogram(show_all=show_all)
except Exception as e:
    print(e)