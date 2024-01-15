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
BATCH_SIZE = 256

# Model path
checkpoint_path = "../Trained_models/CANINE_Multilingual_CANINE_sample_size_10_lr_2e-05_batch_size_256"
model_name = "CANINE_Multilingual_CANINE_sample_size_10_lr_2e-05_batch_size_256"

# %% Import necessary modules
import torch
from n103_Model_loader import Finetuned_model
from n102_DataLoader import Load_val
# from n001_Model_assets import load_model_from_checkpoint, load_tokenizer, update_tokenizer, CANINEOccupationClassifier
# from n102_DataLoader import Load_val, Concat_string, Concat_string_canine

#%% Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% Load data
key, df, df_bin = Load_val(
    model_domain = MODEL_DOMAIN,
    sample_size = 5, # SAMPLE_SIZE
    toyload = True
    )

# %% Load model
model = Finetuned_model(model_name, device = device, batch_size = BATCH_SIZE)

# %% Proof of concept
model.predict(["he has a famr and is the taylor of fine dresses for all the ladys"], what = 5)

# %% Predict 
# Get predictions. The following returns 
preds_w_lang = model.predict(df["occ1"].tolist(), lang = df["lang"].tolist(), what = "probs")
preds_wo_lang = model.predict(df["occ1"].tolist(), what = "probs")

