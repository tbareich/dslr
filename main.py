from dslr import DSLR

dl = DSLR.read_csv("datasets/dataset_train.csv")
dl.show_histogram(ncols=3)
