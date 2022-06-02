from __future__ import annotations
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math

sns.set(font_scale=0.8)
sns.set_style("ticks")
sns.color_palette("bright")


class DataScience:

    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.data = dataframe

    def describe(self) -> pd.DataFrame:
        df = self.data.select_dtypes(include=['int64', 'float64'])
        describe_df = pd.DataFrame(
            index=["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"])

        for column_name in df:
            column = df[column_name][df[column_name].notna()].sort_values()
            mean = column.sum() / column.size
            std = np.sqrt(((column - mean)**2).sum() / (column.size - 1))
            describe_df[column_name] = [
                column.size, mean, std, column.iloc[0],
                self._get_quartile(column, 1),
                self._get_quartile(column, 2),
                self._get_quartile(column, 3), column.iloc[column.size - 1]
            ]
        return describe_df

    def show_histogram(self,
                       ncols: int = 4,
                       figsize=(16, 12),
                       show_all=False,
                       kde=False,
                       features=[]):
        if show_all == False:
            df = self.data[[
                "Hogwarts House", "Arithmancy", "Care of Magical Creatures",
                "Potions"
            ]]
        else:
            df = self.data.drop(
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
            sns.scatterplot(data=self.data,
                            x=self.data["Astronomy"],
                            y=self.data["Defense Against the Dark Arts"],
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
            df = self.data.drop([
                "First Name", "Last Name", "Birthday", "Best Hand", "Index",
                "Arithmancy", "Defense Against the Dark Arts", "Potions",
                "Care of Magical Creatures"
            ],
                                axis=1)
        else:
            df = self.data.drop(
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
            plt.subplots_adjust(left=0.04,
                                bottom=0.05,
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
    def read_csv(cls, path: str) -> DataScience:
        data = pd.read_csv(path)
        return cls(dataframe=data)

    def _get_quartile(self, column, q_quartile) -> float:
        # quartile position
        qu_p = q_quartile * .25 * (column.size - 1)
        # quartile index
        qu_i = int(qu_p)
        # quartile position fraction part
        qu_fract = qu_p - int(qu_i)
        quartile = column.iloc[qu_i] * (
            1 - qu_fract) + column.iloc[qu_i + 1] * qu_fract
        return quartile
