# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 11:19:36 2024

@author: christian-vs
"""


import argparse
from histocc.adversarial_occupations import translated_strings_wrapper, generate_adversarial_wrapper, generate_random_strings_wrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data-path', type=str, default="Data/Training_data")
    parser.add_argument('--storage-path', type=str, default="Data/Adversarial_data")
    
    parser.add_argument('--n-max', type=int, default=20)
    parser.add_argument('--n-trans', type=int, default=1)
    
    parser.add_argument('--num-adv-simple', type=int, default=100000)
    parser.add_argument('--num-adv-double-trans', type=int, default=50000)
    parser.add_argument('--num-trans', type=int, default=100000)
    parser.add_argument('--num-rand', type=int, default=100000)
    
    parser.add_argument('--toyload', type=bool, default=False) # Can be used to load some data quickly for debugging
    
    args = parser.parse_args()
    
    return args


def main():
    args = parse_args()
    
    # Using attacker.attack to generates adv. examples:
    generate_adversarial_wrapper(
        data_path=args.data_path,
        storage_path=args.storage_path,
        toyload=args.toyload,
        double_translate=False, 
        sample_size = args.num_adv_simple, 
        n_max = args.n_max, 
        n_trans = args.n_trans
        )

    # Using double translation to generates adv. examples (takes longer):
    generate_adversarial_wrapper(
        data_path=args.data_path,
        storage_path=args.storage_path,
        toyload=args.toyload, 
        double_translate=True, 
        sample_size = args.num_adv_double_trans, 
        n_max = args.n_max, 
        n_trans = args.n_trans
        )

    # Generating strings in every language (takes longer)
    translated_strings_wrapper(
        toyload=args.toyload,
        data_path=args.data_path,
        storage_path=args.storage_path,
        sample_size = args.num_trans
        )

    # Generating random strings
    generate_random_strings_wrapper(
        storage_path=args.storage_path,
        num_strings=args.num_rand,
        )

if __name__ == '__main__':
    main()