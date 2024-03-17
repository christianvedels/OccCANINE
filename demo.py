# -*- coding: utf-8 -*-
"""
Created on 2024-01-15

@author: christian-vs

Prediction for applications

SETUP:
    - See readme2.md
"""


import argparse

from histocc import OccCANINE, DATASETS


def parse_args() -> argparse.Namespace:
    _default_example_strs = [
        ["tailor of the finest suits"],
        ["the train's fireman"],
        ["nurse at the local hospital"],
    ]

    parser = argparse.ArgumentParser()

    parser.add_argument('--examples', type=str, nargs='+', default=None)
    parser.add_argument('--toy-dataset-fn-out', type=str, default=None)

    parser.add_argument('--language', type=str, default='en') # TODO add arg choices based on supported languages
    parser.add_argument('--threshold', type=float, default=0.22) # Best F1 for English

    args = parser.parse_args()

    if args.examples is None:
        args.examples = _default_example_strs

    return args


def main():
    args = parse_args()

    # Load model
    model = OccCANINE()

    # Loop through single-str example
    for example_str in args.examples:
        occ_code, prob, occ = model.predict(
            example_str,
            lang=args.language,
            get_dict=True,
            threshold=args.threshold,
            )[0][0]

        print(f'HISCO code: {occ_code}. Occupation: {occ}. Certainty: {prob * 100:.2f}%')

    # Predict on toy dataset if filename for output specified
    if args.toy_dataset_fn_out is None:
        return

    print('--toy-dataset-fn-out specified -- predicting codes for toy data')
    data = DATASETS['toydata']()
    model.verbose = True # Set updates to True

    model_prediction = model.predict(
        data["occ1"],
        lang=args.language,
        threshold=args.threshold,
        )

    model_prediction["occ1"] = data["occ1"]

    print(f'Writing output to {args.toy_dataset_fn_out}')
    model_prediction.to_csv(args.toy_dataset_fn_out, index=False)


if __name__ == '__main__':
    main()
