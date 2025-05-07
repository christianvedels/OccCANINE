# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 11:19:36 2024

@author: christian-vs

"""

import argparse

from histocc.prediction_assets import OccCANINE

from histocc.adversarial_occupations import (
    translated_strings_wrapper,
    generate_adversarial_wrapper,
    generate_random_strings_wrapper,
    lang_mapping
)

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-path', type=str, default="Data/Training_data")
    parser.add_argument('--file', type=str, default=None)  # Path to file with data to be translated. If None, the script will use the data_path and load all files in it.
    parser.add_argument('--storage-path', type=str, default="Data/Adversarial_data_other")

    parser.add_argument('--n-max', type=int, default=20)
    parser.add_argument('--n-trans', type=int, default=1)

    parser.add_argument('--num-adv-simple', type=int, default=200000)
    parser.add_argument('--num-adv-double-trans', type=int, default=100000)
    parser.add_argument('--num-trans', type=int, default=200000)
    parser.add_argument('--num-rand', type=int, default=100000)

    parser.add_argument('--toyload', type=bool, default=False)  # Can be used to load some data quickly for debugging

    parser.add_argument('--task', type=str, default=None, choices=['2xtranslate', 'attack', 'translate', 'gibberish'])

    parser.add_argument('--lang', type=str, default=None)
    parser.add_argument('--no-occ-labels', type=str, nargs="*", default=[-1])

    parser.add_argument('--model-path', default="OccCANINE_s2s_mix")
    parser.add_argument('--model-local', type=bool, default=False)
    parser.add_argument('--model-system', type=str, default="hisco")
    parser.add_argument('--model-use-within-block-sep', type=bool, default=False)

    # Tmp while developing:
    # parser.add_argument('--model-path', default="D:/Dropbox/Research_projects/HISCO/OccCANINE/Data/models/mixer-psti-ft/last.bin")
    # parser.add_argument('--model-local', type=bool, default=True)
    # parser.add_argument('--model-use-within-block-sep', type=bool, default=True)
    # parser.add_argument('--model-system', type=str, default="PSTI")
    # parser.add_argument('--file', type=str, default="Data/Training_data_other/EN_PSTI_CAMPOP_n_unq1000_train.csv")
    # parser.add_argument('--num-adv-simple', type=int, default=200)
    # parser.add_argument('--num-adv-double-trans', type=int, default=200)
    # parser.add_argument('--num-trans', type=int, default=200)
    # parser.add_argument('--num-rand', type=int, default=200)

    parser.add_argument('--class-balance', type=bool, default=True)

    parser.add_argument('--methods', type=str, nargs='+', default=['attack', '2xtranslate', 'translate', 'gibberish'],
                        help="Specify which methods to run: 'attack', '2xtranslate', 'translate', 'gibberish'")

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # Load model
    model = OccCANINE(args.model_path, hf=not args.model_local, system=args.model_system, use_within_block_sep=args.model_use_within_block_sep)

    # Langs
    if args.lang:
        langs = [args.lang]
    else:
        langs = lang_mapping.keys()

    if args.file:
        data_path = args.file
    else:
        data_path = args.data_path

    # Execute selected methods
    if 'attack' in args.methods:
        # Using attacker.attack to generate adv. examples:
        res1 = generate_adversarial_wrapper(
            data_path=data_path,
            hisco_predictor=model,
            toyload=args.toyload,
            double_translate=False,
            sample_size=args.num_adv_simple,
            n_max=args.n_max,
            n_trans=args.n_trans,
            class_balance=args.class_balance
        )
        res1["Adversarial_method"] = "attack"
        # res1.to_csv(args.storage_path + "/adversarial_examples_double_translateFalse.csv", index=False)

    if '2xtranslate' in args.methods:
        # Using double translation to generate adv. examples (takes longer):
        res2 = generate_adversarial_wrapper(
            data_path=data_path,
            hisco_predictor=model,
            toyload=args.toyload,
            double_translate=True,
            sample_size=args.num_adv_simple,
            n_max=args.n_max,
            n_trans=args.n_trans,
            class_balance=args.class_balance
        )
        res2["Adversarial_method"] = "2xtranslate"
        # res2.to_csv(args.storage_path + "/adversarial_examples_double_translateTrue.csv", index=False)

    if 'translate' in args.methods:
        # Generating strings in every language (takes longer)
        res3 = translated_strings_wrapper(
            data_path=data_path,
            toyload=args.toyload,
            sample_size=args.num_trans
        )
        res3["Adversarial_method"] = "translate"
        # res3.to_csv(args.storage_path + "/translated_strings.csv", index=False)

    if 'gibberish' in args.methods:
        # Generating random strings
        res4 = generate_random_strings_wrapper(
            num_strings=args.num_rand,
            system=model.system,
            no_occ_labels=args.no_occ_labels,
            langs=langs
        )
        res4["Adversarial_method"] = "gibberish"
        # res4.to_csv(args.storage_path + "/random_strings.csv", index=False)
    

    # Merge all results into one file
    if 'attack' in args.methods or '2xtranslate' in args.methods or 'translate' in args.methods or 'gibberish' in args.methods:
        all_results = []
        if 'attack' in args.methods:
            all_results.append(res1)
        if '2xtranslate' in args.methods:
            all_results.append(res2)
        if 'translate' in args.methods:
            all_results.append(res3)
        if 'gibberish' in args.methods:
            all_results.append(res4)

        merged_results = pd.concat(all_results, ignore_index=True)
        merged_results.to_csv(args.storage_path + "/merged_results.csv", index=False)
        print(f"Merged results saved to {args.storage_path}/merged_results.csv")

if __name__ == '__main__':
    main()
