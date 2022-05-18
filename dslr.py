import numpy as np
import pandas as pd


class DSLR:

    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.dataframe = dataframe
        pass

    def describe(self) -> pd.DataFrame:
        df = self.dataframe
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

    @staticmethod
    def read_csv(path):
        try:
            df = pd.read_csv(path)
            return DSLR(dataframe=df)
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
