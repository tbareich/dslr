from dslr import DSLR

dl = DSLR.read_csv("datasets/dataset_train.csv")
dl.show_scatter_plot()
dl.show_histogram()
dl.show_pair_plot()
