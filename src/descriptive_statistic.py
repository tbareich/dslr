# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    descriptive_statistic.py                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: tbareich <tbareich@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2022/06/06 14:29:40 by tbareich          #+#    #+#              #
#    Updated: 2022/06/07 08:54:36 by tbareich         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from __future__ import annotations
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math

from src.statistics import Statistics

sns.set(font_scale=0.8)
sns.set_style("ticks")
sns.color_palette("bright")


class DescriptiveStatistic:

    def __init__(self, data: pd.DataFrame) -> None:
        self._data = data

    def describe(self) -> pd.DataFrame:
        df = self._data.select_dtypes(include=['int64', 'float64'])
        describe_df = pd.DataFrame(
            index=["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"])
        for column_name in df:
            column = df[column_name][df[column_name].notna()].sort_values()
            X = column.values
            mean = Statistics.mean(X)
            std = Statistics.std(X, mean)
            count = len(X)
            describe_df[column_name] = [
                count,
                mean,
                std,
                X[0],
                Statistics.percentil(X, 1),
                Statistics.percentil(X, 2),
                Statistics.percentil(X, 3),
                X[count - 1],
            ]
        return describe_df

    def show_histogram(
            self,
            ncols: int = 4,
            figsize=(16, 12),
            show_all=False,
            kde=False,
    ):
        if show_all == False:
            df = self._data[[
                "Hogwarts House", "Arithmancy", "Care of Magical Creatures",
                "Potions"
            ]]
        else:
            df = self._data.drop(
                ["First Name", "Last Name", "Birthday", "Best Hand", "Index"],
                axis=1)
        cols_len = df.shape[1] - 1
        ncols = math.ceil(cols_len /
                          ncols) if cols_len / ncols > 1 else cols_len
        nrows = ncols if cols_len / ncols > 1 else 1
        figsize = (12, 5) if nrows == 1 else figsize
        fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
        sns.despine()
        x = 0
        y = 0
        for column_name in df:
            if column_name == "Hogwarts House":
                continue
            ax = axs[x] if nrows == 1 else axs[y, x]
            hist = sns.histplot(data=df,
                                x=column_name,
                                ax=ax,
                                hue="Hogwarts House",
                                alpha=0.6,
                                linewidth=0,
                                kde=kde)
            hist.set_xlabel(column_name.replace(' ', '\n', 2))
            x += 1
            if x == ncols:
                x = 0
                y += 1
        for i in range(x, ncols):
            for j in range(y, nrows):
                fig.delaxes(axs[j][i])
        if show_all == False:
            fig.suptitle(
                "Hogwarts courses with homogeneous score distribution",
                fontsize=16)
        plt.tight_layout()
        plt.show()

    def show_scatter_plot(self, show_all=False):
        if show_all == False:
            fig = plt.figure()
            sns.scatterplot(data=self._data,
                            x=self._data["Astronomy"],
                            y=self._data["Defense Against the Dark Arts"],
                            linewidth=0,
                            s=10,
                            hue="Hogwarts House")
            fig.suptitle("Two similar features", fontsize=16)
            plt.tight_layout()
            plt.show()
        else:
            self.show_pair_plot(show_all)

    def show_pair_plot(self, show_all=False, figsize=(18, 12)):
        show_legend = True
        if show_all == False:
            df = self._data.drop([
                "First Name", "Last Name", "Birthday", "Best Hand", "Index",
                "Arithmancy", "Astronomy", "Potions",
                "Care of Magical Creatures"
            ],
                                 axis=1)
        else:
            df = self._data.drop(
                ["First Name", "Last Name", "Birthday", "Best Hand", "Index"],
                axis=1)

        columns_len = df.shape[1]
        fig, axs = plt.subplots(columns_len - 1,
                                columns_len - 1,
                                figsize=figsize)
        sns.despine()
        y = 0
        for column_y in df:
            x = 0
            if column_y == "Hogwarts House":
                continue
            for column_x in df:
                if column_x == "Hogwarts House":
                    continue
                if (column_x == column_y):
                    plot = sns.histplot(
                        data=df,
                        x=column_x,
                        ax=axs[y, x],
                        hue="Hogwarts House",
                        legend=False,
                        alpha=0.6,
                        linewidth=0,
                    )
                else:
                    plot = sns.scatterplot(data=df,
                                           x=column_x,
                                           y=column_y,
                                           ax=axs[y, x],
                                           hue="Hogwarts House",
                                           legend=show_legend,
                                           alpha=0.5,
                                           linewidth=0,
                                           s=7)
                    if show_legend:
                        plot.legend(bbox_to_anchor=(1, 0.5),
                                    loc="center right",
                                    bbox_transform=fig.transFigure)
                        show_legend = False
                if show_all == True:
                    plot.axes.xaxis.set_ticks([])
                    plot.axes.yaxis.set_ticks([])
                if x != 0:
                    plot.set(ylabel=None)
                else:
                    plot.set(ylabel=column_y.replace(' ', '\n', 2))
                if y != columns_len - 2:
                    plot.set(xlabel=None)
                else:
                    plot.set(xlabel=column_x.replace(' ', '\n', 2))
                x += 1
            y += 1
        if show_all == False:
            fig.suptitle("Pair plot of the remaining courses", fontsize=16)
            plt.subplots_adjust(left=0.06,
                                bottom=0.07,
                                right=0.93,
                                top=0.95,
                                hspace=0.4,
                                wspace=0.4)
        else:
            plt.subplots_adjust(left=0.04,
                                bottom=0.05,
                                right=0.93,
                                top=0.97,
                                wspace=0.1,
                                hspace=0.1)
        plt.show()

    @classmethod
    def read_csv(cls, path: str) -> DescriptiveStatistic:
        data = pd.read_csv(path)
        return cls(data=data)
