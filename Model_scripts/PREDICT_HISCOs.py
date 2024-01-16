# -*- coding: utf-8 -*-
"""
Created on 2024-01-15

@author: christian-vs

Prediction for applications
"""

import os
script_directory = os.path.dirname(os.path.abspath(__name__))
os.chdir(script_directory)

# %%
# Input / output relative to "Model_scripts"
file_path = "../Deliveries/230522 Human captial in the Nordics/Occupations_dk.csv"
output_path = "../Deliveries/230522 Human captial in the Nordics/Predictions"

# Choose which GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MODEL_DOMAIN = "Multilingual_CANINE" # Type of model
BATCH_SIZE = 256 # Adjust this down if GPU cannot handle it. Adjust it up otherwise

# Name of the finetuned model to use (must be located in "Trained_models") 
model_name = "CANINE_Multilingual_CANINE_sample_size_10_lr_2e-05_batch_size_256"

# %% Import necessary modules
import torch
import pandas as pd
from n103_Prediction_assets import Finetuned_model
from n102_DataLoader import Load_val

#%% Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% Load data
df = pd.read_csv(file_path)

# %% Load model
model = Finetuned_model(model_name, device = device, batch_size = BATCH_SIZE)

# %% Proof of concept
model.predict(["he has a famr and is the taylor of fine dresses for all the ladys"], what = 5)

# %% Predict 
# Get predictions. The following returns 
preds_w_lang = model.predict(df["occ1"].tolist(), lang = df["lang"].tolist(), what = "probs")
preds_wo_lang = model.predict(df["occ1"].tolist(), what = "probs")

    