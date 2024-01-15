# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 13:30:09 2024

@author: Christian Vedel 
"""

import os
script_directory = os.path.dirname(os.path.abspath(__name__))
os.chdir(script_directory)

# %%
# Choose which GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MODEL_DOMAIN = "Multilingual_CANINE" # Type of model
MODEL_SIZE = "base"

# Model path
# checkpoint_path = "../Trained_models/CANINE_Multilingual_CANINE_sample_size_10_lr_2e-05_batch_size_256"
model_name = "CANINE_Multilingual_CANINE_sample_size_10_lr_2e-05_batch_size_256"



# %% Import necessary modules
import torch
from n001_Model_assets import load_model_from_checkpoint, load_tokenizer, update_tokenizer, CANINEOccupationClassifier
from n102_DataLoader import Load_val, Concat_string, Concat_string_canine

#%% Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% Load data
key, df, df_bin = Load_val(
    model_domain = MODEL_DOMAIN,
    sample_size = 5, # SAMPLE_SIZE
    toyload = True
    )

# %% Input string
class ConcatString:
    def __init__(self, model_domain):
        self.model_domain = model_domain
        
    def get(self, occ1, lang):
        if self.model_domain == "Multilingual":
            return Concat_string(occ1, lang)
        elif self.model_domain == "Multilingual_CANINE":
            return Concat_string_canine(occ1, lang)
        else: 
            raise Exception("Not implemented")

concat_string = ConcatString(MODEL_DOMAIN)

# %% Concat strings
df['concat_string0'] = [concat_string.get(occ1, lang) for occ1, lang in zip(df['occ1'].tolist(), df['lang'].tolist())]
df['concat_string1'] = [concat_string.get(occ1, 'unk') for occ1 in df['occ1'].tolist()]

# %% Tokenizer
tokenizer = load_tokenizer(
    model_domain = MODEL_DOMAIN,
    model_size = MODEL_SIZE
    )

if MODEL_DOMAIN != "Multilingual_CANINE":
    tokenizer = update_tokenizer(tokenizer, df)

# %% Load model
model = CANINEOccupationClassifier(
    model_domain = MODEL_DOMAIN,
    n_classes = len(key), 
    tokenizer = tokenizer, 
    dropout_rate = 0
    )
model.to(device)

# %% Load model checkpoint
model, tokenizer = load_model_from_checkpoint(checkpoint_path, model, MODEL_DOMAIN)


