from classes.ds import DS
import argparse

try:
    parser = argparse.ArgumentParser(description='show data description.')
    parser.add_argument('filename',
                        type=str,
                        help='the file containing the training data.')

    args = parser.parse_args()
    filename = args.filename

    ds = DS.read_csv(filename)
    print(ds.describe())
except Exception as e:
    print(e)
