# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 16:22:18 2024

@author: christian-vs
"""

import pandas as pd
import torch

from .model_assets import CANINEOccupationClassifier_hub

NAME = "CANINE_Multilingual_CANINE_sample_size_10_lr_2e-05_batch_size_256"

def main(): # FIXME what is the purpose of this fn?
    key = pd.read_csv("Data/Key.csv") # Load key and convert to dictionary
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

    loaded_state = torch.load("Model/" + NAME + ".bin")
    model.load_state_dict(loaded_state)

    model.save_pretrained("Model/OccCANINE_forHF", config=config)

    # Test
    model = CANINEOccupationClassifier_hub.from_pretrained(
        "revert94/OccCANINE",
        force_download = True
        )
