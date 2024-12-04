# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 16:22:18 2024

@author: christian-vs
"""


import argparse

import torch

from histocc import DATASETS
from histocc.model_assets import CANINEOccupationClassifier_hub, Seq2SeqMixerOccCANINE_hub, Seq2SeqOccCANINE_hub

from histocc.formatter import (
    hisco_blocky5,
    BOS_IDX,
)


NAME = "CANINE_Multilingual_CANINE_sample_size_10_lr_2e-05_batch_size_256"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('--fn-in', type=str, default=f'./Model/{NAME}.bin')
    parser.add_argument('--fn-in-mixer', type=str, default=f'Data/models/mixer-s2s=10-s=1605000.bin')
    parser.add_argument('--fn-in-s2s', type=str, default=f'Data/models/baseline-s=1600000.bin')
    parser.add_argument('--fn-out', type=str, default='Data/models/OccCANINE_forHF')
    parser.add_argument('--fn-out-mixer', type=str, default='Data/models/Seq2SeqMixerOccCANINE_forHF')
    parser.add_argument('--fn-out-s2s', type=str, default='Data/models/Seq2SeqOccCANINE_forHF')

    parser.add_argument('--test', action='store_true', default=False)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # Load formatter to use dims from it
    formatter = hisco_blocky5()

    # Load keys
    key = DATASETS['keys']()
    key = key[1:]
    key = zip(key.code, key.hisco)
    key = list(key)

    config = {
        "model_domain": "Multilingual_CANINE",
        "n_classes": len(key),
        "num_classes": formatter.num_classes,
        "dropout_rate": 0,
        "model_type": "canine"
    }

    # # In training PyTorchModelHubMixin was missing
    # model = CANINEOccupationClassifier_hub(config)

    # loaded_state = torch.load(args.fn_in)
    # model.save_pretrained(args.fn_out, config=config)

    # Repeat for Seq2SeqMixerOccCANINE_hub
    config["model_type"] = "seq2seq_mixer"
    config["num_classes_flat"] = config["n_classes"]+1
    model = Seq2SeqMixerOccCANINE_hub(config)

    loaded_state = torch.load(args.fn_in_mixer)
    model.load_state_dict(loaded_state['model'])

    model.save_pretrained(args.fn_out_mixer, config=config)

    if args.test:
        model = Seq2SeqMixerOccCANINE_hub.from_pretrained(
            "christianvedel/Seq2SeqMixerOccCANINE",
            force_download=True,
            )

    # Repeat for Seq2SeqOccCANINE_hub
    config["model_type"] = "seq2seq"
    model = Seq2SeqOccCANINE_hub(config)

    loaded_state = torch.load(args.fn_in_s2s)
    model.load_state_dict(loaded_state)

    model.save_pretrained(args.fn_out_s2s, config=config)

    if args.test:
        model = Seq2SeqOccCANINE_hub.from_pretrained(
            "christianvedel/Seq2SeqOccCANINE",
            force_download=True,
            )

    model.save_pretrained(args.fn_out, config=config)

    # Test
    if args.test:
        model = CANINEOccupationClassifier_hub.from_pretrained(
            "christianvedel/OccCANINE",
            force_download=True,
            )

    # Repeat for Seq2SeqMixerOccCANINE_hub
    config["model_type"] = "seq2seq_mixer"
    config["num_classes"] = [len(key)]  # Assuming num_classes is a list of class counts
    config["num_classes_flat"] = len(key)  # Assuming num_classes_flat is the total number of classes
    model = Seq2SeqMixerOccCANINE_hub(config)

    loaded_state = torch.load(args.fn_in)
    model.load_state_dict(loaded_state)

    model.save_pretrained(args.fn_out, config=config)

    if args.test:
        model = Seq2SeqMixerOccCANINE_hub.from_pretrained(
            "christianvedel/Seq2SeqMixerOccCANINE",
            force_download=True,
            )

    # Repeat for Seq2SeqOccCANINE_hub
    config["model_type"] = "seq2seq"
    config["num_classes"] = [len(key)]  # Assuming num_classes is a list of class counts
    model = Seq2SeqOccCANINE_hub(config)

    loaded_state = torch.load(args.fn_in)
    model.load_state_dict(loaded_state)

    model.save_pretrained(args.fn_out, config=config)

    if args.test:
        model = Seq2SeqOccCANINE_hub.from_pretrained(
            "christianvedel/Seq2SeqOccCANINE",
            force_download=True,
            )


if __name__ == '__main__':
    main()
