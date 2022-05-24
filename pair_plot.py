from classes.dslr import DSLR
import argparse

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

dslr_obj = DSLR.read_csv(filename)

dslr_obj.show_pair_plot(show_all=show_all)
