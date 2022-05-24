from classes.dslr import DSLR
import argparse

parser = argparse.ArgumentParser(description='show data description.')
parser.add_argument('filename',
                    type=str,
                    help='the file containing the training data.')

args = parser.parse_args()
filename = args.filename

dslr = DSLR.read_csv(filename)
print(dslr.describe())
