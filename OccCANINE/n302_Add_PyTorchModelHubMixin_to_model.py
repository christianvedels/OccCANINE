# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 16:22:18 2024

@author: christian-vs
"""

#%%
import os
script_directory = os.path.dirname(os.path.abspath(__name__))
os.chdir(script_directory)

# %% Import modules
from n001_Model_assets import CANINEOccupationClassifier_hub
from n103_Prediction_assets import Get_adapated_tokenizer
import pandas as pd
import torch

# %%
name = "CANINE_Multilingual_CANINE_sample_size_10_lr_2e-05_batch_size_256"

key = pd.read_csv("Data/Key.csv") # Load key and convert to dictionary
key = key[1:]
key = zip(key.code, key.hisco)
key = list(key)

tokenizer = Get_adapated_tokenizer(name)

config = {
    "model_domain": "Multilingual_CANINE",
    "n_classes": len(key),
    "dropout_rate": 0,
    "model_type": "canine"
}

# In training PyTorchModelHubMixin was missing
model = CANINEOccupationClassifier_hub(config)

loaded_state = torch.load("Model/"+name+".bin")
model.load_state_dict(loaded_state)

model.save_pretrained("Model/OccCANINE_forHF", config=config)

# %% Test
model = CANINEOccupationClassifier_hub.from_pretrained(
    "revert94/OccCANINE",
    force_download = True
    )
