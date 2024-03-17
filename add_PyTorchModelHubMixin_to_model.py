# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 16:22:18 2024

@author: christian-vs
"""


import argparse

import torch

from histocc import DATASETS, CANINEOccupationClassifier_hub


NAME = "CANINE_Multilingual_CANINE_sample_size_10_lr_2e-05_batch_size_256"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('--fn-in', type=str, default=f'./Model/{NAME}.bin')
    parser.add_argument('--fn-out', type=str, default='./Model/OccCANINE_forHF')

    parser.add_argument('--test', action='store_true', default=False)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # Load keys
    key = DATASETS['keys']()
    key = key[1:]
    key = zip(key.code, key.hisco)
    key = list(key)

    config = {
        "model_domain": "Multilingual_CANINE",
        "n_classes": len(key),
        "dropout_rate": 0,
        "model_type": "canine"
    }

    # In training PyTorchModelHubMixin was missing
    model = CANINEOccupationClassifier_hub(config)

    loaded_state = torch.load(args.fn_in)
    model.load_state_dict(loaded_state)

    model.save_pretrained(args.fn_out, config=config)

    # Test
    if args.test:
        model = CANINEOccupationClassifier_hub.from_pretrained(
            "revert94/OccCANINE",
            force_download=True,
            )
