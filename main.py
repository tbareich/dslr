from dslr import DSLR

dl = DSLR.read_csv("datasets/dataset_train.csv")

print(dl.describe())
