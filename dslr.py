from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math


class DSLR:

    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.data = dataframe

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

    def show_histogram(self, ncols: int = 3, figsize=(12, 12)):
        sns.set_style("whitegrid")
        sns.set(font_scale=0.8)
        df = self.data.select_dtypes(include=['float64'])
        fig, axs = plt.subplots(math.ceil(df.shape[1] / ncols),
                                ncols,
                                figsize=figsize)
        x = 0
        y = 0
        for column_name in df:
            p = sns.histplot(data=df, x=column_name, ax=axs[y, x])
            p.set_ylabel("Frequency")
            x += 1
            if x == ncols:
                x = 0
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
