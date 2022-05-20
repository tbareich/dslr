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
        sns.set_palette('RdBu_r')

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

    def show_histogram(self, figsize=(25, 10)):
        fig, axs = plt.subplots(4, 13, figsize=figsize)
        sns.despine()
        houses = ["Ravenclaw", "Slytherin", "Gryffindor", "Hufflepuff"]
        y = 0
        describ_df = self.describe()
        # print(describ_df)
        for h in houses:
            x = 0
            house_df = self.data[self.data["Hogwarts House"] ==
                                 h].select_dtypes(include=['float64'])
            for column_name in house_df:
                house_df[column_name] = (house_df[column_name] -
                                         describ_df[column_name]['Min']) / (
                                             describ_df[column_name]['Max'] -
                                             describ_df[column_name]['Min'])
                s = sns.histplot(
                    data=house_df,
                    x=column_name,
                    ax=axs[y, x],
                    stat='count',
                )
                if x != 0:
                    s.set_ylabel('')
                else:
                    s.set_ylabel(h)
                x += 1
            y += 1
        plt.tight_layout()
        plt.show()

    def show_scatter(self):
        g = sns.PairGrid(
            self.data,
            vars=[
                'Arithmancy', 'Astronomy', 'Herbology',
                'Defense Against the Dark Arts', 'Divination',
                'Muggle Studies', 'Ancient Runes', 'History of Magic',
                'Transfiguration', 'Potions', 'Care of Magical Creatures',
                'Charms', 'Flying'
            ],
            hue="Hogwarts House",
        )

        g.map_offdiag(sns.scatterplot)
        g.add_legend()
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
