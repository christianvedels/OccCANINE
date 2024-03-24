"""
CLI for predicting HISCO codes based on .csv-file.

Example use:
    1) python predict.py --fn-in path/to/input/data.csv --col occ1 --fn-out path/to/output/data.csv
    2) python predict.py --fn-in path/to/input/data.csv --col occ1 --fn-out path/to/output/data.csv --language en

"""


import argparse
import os

import pandas as pd

from histocc import OccCANINE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('--fn-in', type=str, required=True)
    parser.add_argument('--fn-out', type=str, required=True)

    parser.add_argument('--col', type=str, default=None)

    parser.add_argument('--language', type=str, default='en') # TODO add arg choices based on supported languages
    parser.add_argument('--threshold', type=float, default=0.22) # Best F1 for English
    parser.add_argument('--non-verbose', action='store_true', default=False)

    args = parser.parse_args()

    if not os.path.isfile(args.fn_in):
        raise FileNotFoundError(f'--fn-in {args.fn_in} does not exist')

    if os.path.isfile(args.fn_out):
        raise FileExistsError(f'--fn-out {args.fn_out} already exists')

    return args


def main():
    args = parse_args()

    # Load model
    model = OccCANINE()
    model.verbose = not args.non_verbose

    # Load input data
    data = pd.read_csv(args.fn_in)
    col = args.col if args.col is not None else data.columns[0]

    # Predict HISCO codes
    model_prediction = model.predict(
        data[col],
        lang=args.language,
        threshold=args.threshold,
        )

    print(f'Writing output to {args.fn_out}')
    model_prediction.to_csv(args.fn_out, index=False)


if __name__ == '__main__':
    main()
