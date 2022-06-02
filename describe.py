from core.data_science import DataScience
import argparse

parser = argparse.ArgumentParser(description='show data description.')
parser.add_argument('filename',
                    type=str,
                    help='the file containing the training data.')

args = parser.parse_args()
filename = args.filename

ds = DataScience.read_csv(filename)
print(ds.describe())
