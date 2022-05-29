import argparse
from classes.logistic_regression import LogisticRegression

def filter_data(df):
    df = df.drop()

try:
    parser = argparse.ArgumentParser(description='train you model.')

    parser.add_argument('filename',
                        type=str,
                        help='the file containing the training data.')

    args = parser.parse_args()
    filename = args.filename

    lr = LogisticRegression.read_csv(filename)
except Exception as e:
    print(e)
