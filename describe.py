from src.core import Core
import argparse

try:
    parser = argparse.ArgumentParser(description='show data description.')
    parser.add_argument('filename',
                        type=str,
                        help='the file containing the training data.')

    args = parser.parse_args()
    filename = args.filename

    ds = Core.read_csv(filename)
    print(ds.describe())
except Exception as e:
    print(e)