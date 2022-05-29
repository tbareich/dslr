from __future__ import annotations
import pandas as pd


class LogisticRegression:

    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.data = dataframe

    def fit(self) -> None:
        pass

    def predict(self) -> None:
        pass

    @classmethod
    def read_csv(cls, path: str) -> LogisticRegression:
        try:
            data = pd.read_csv(path)
            return cls(dataframe=data)
        except OSError as e:
            raise Exception(f'{path} file doesn\'t exist.')
        except Exception:
            raise Exception('Something went wrong.')
