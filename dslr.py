from __future__ import annotations
from pydoc import describe
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math


class DSLR:

    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.data = dataframe
        sns.set(font_scale=0.8)
        sns.set_style("ticks")

    def describe(self) -> pd.DataFrame:
        df = self.data
        df = df.select_dtypes(include=['int64', 'float64'])
        describe_df = pd.DataFrame(
            index=["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"])

        for column_name in df:
            column = df[column_name][df[column_name].notna()].sort_values()
            mean = column.sum() / column.size
            std = np.sqrt(((column - mean)**2).sum() / (column.size - 1))
            describe_df[column_name] = [
                column.size, mean, std, column.iloc[0],
                self.__get_quartile(column, 1),
                self.__get_quartile(column, 2),
                self.__get_quartile(column, 3), column.iloc[column.size - 1]
            ]
        return describe_df

    def show_histogram(self, ncols: int = 4, figsize=(14, 10)):
        df = self.data.drop(
            ["First Name", "Last Name", "Birthday", "Best Hand", "Index"],
            axis=1)
        fig, axs = plt.subplots(math.ceil(df.shape[1] / ncols),
                                ncols,
                                figsize=figsize)
        sns.despine()
        x = 0
        y = 0
        for column_name in df:
            if column_name == "Hogwarts House":
                continue
            p = sns.histplot(data=df,
                             x=column_name,
                             ax=axs[y, x],
                             hue="Hogwarts House")
            x += 1
            if x == ncols:
                x = 0
                y += 1
        plt.tight_layout()
        plt.show()

    def show_scatter(self):
        df = self.data.drop([
            "First Name", "Last Name", "Birthday", "Best Hand", "Index",
            "Arithmancy", "Astronomy", "Potions", "Care of Magical Creatures"
        ],
                            axis=1)

        columns_len = df.shape[1]
        fig, axs = plt.subplots(columns_len - 1,
                                columns_len - 1,
                                figsize=(15 + 3, 10 + 3))
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
                    ax = sns.histplot(data=df,
                                      x=column_x,
                                      ax=axs[y, x],
                                      hue="Hogwarts House",
                                      legend=False)
                else:
                    ax = sns.scatterplot(data=df,
                                         x=column_x,
                                         y=column_y,
                                         ax=axs[y, x],
                                         hue="Hogwarts House",
                                         legend=False,
                                         alpha=0.5,
                                         linewidth=0,
                                         s=8)
                if x != 0:
                    ax.set(ylabel=None)
                if y != columns_len - 2:
                    ax.set(xlabel=None)
                x += 1
            y += 1
        plt.tight_layout()
        plt.show()

    @classmethod
    def read_csv(cls, path: str) -> DSLR:
        try:
            data = pd.read_csv(path)
            return cls(dataframe=data)
        except OSError:
            print('The training file does\'t exist.')
        except Exception:
            print('Something went wrong.')

    def __get_quartile(self, column, q_quartile) -> float:
        # quartile position
        qu_p = q_quartile * .25 * (column.size - 1)
        # quartile index
        qu_i = int(qu_p)
        # quartile position fraction part
        qu_fract = qu_p - int(qu_i)
        quartile = column.iloc[qu_i] * (
            1 - qu_fract) + column.iloc[qu_i + 1] * qu_fract
        return quartile
