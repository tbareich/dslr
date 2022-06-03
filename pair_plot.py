from src.core import DataScience
import argparse

try:
    parser = argparse.ArgumentParser(description='show data pair plot.')

    parser.add_argument('filename',
                        type=str,
                        help='the file containing the training data.')
    parser.add_argument('--show_all',
                        action='store_const',
                        const=True,
                        default=False,
                        help='show all features histogram.')

    args = parser.parse_args()
    filename = args.filename
    show_all = args.show_all

    ds = DataScience.read_csv(filename)

    ds.show_pair_plot(show_all=show_all)
except Exception as e:
    print(e)