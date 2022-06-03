from sklearn.metrics import accuracy_score

import pandas as pd

house = pd.read_csv("out/house.csv")
truth = pd.read_csv("datasets/dataset_truth.csv")

print(
    accuracy_score(truth["Hogwarts House"].values,
                   house["Hogwarts House"].values))
