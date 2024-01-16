# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 13:30:09 2024

@author: Christian Vedel 

Predictions for model evaluation
"""

import os
script_directory = os.path.dirname(os.path.abspath(__name__))
os.chdir(script_directory)

# %%
# Choose which GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MODEL_DOMAIN = "Multilingual_CANINE" # Type of model
BATCH_SIZE = 256

# Show updates when running?
VERBOSE = True

# Where should the files be saved to?
output_dir = "../Data/Predictions/"

# Name of the finetuned model to use (must be located in "Trained_models") 
model_name = "CANINE_Multilingual_CANINE_sample_size_10_lr_2e-05_batch_size_256"

# %% Import necessary modules
import torch
import numpy as np
import pandas as pd

from n103_Prediction_assets import Finetuned_model
from n102_DataLoader import Load_val


#%% Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% Load data
key, df = Load_val(
    model_domain = MODEL_DOMAIN,
    sample_size = 5, # 10^sample_size
    toyload = False
    )

# %% Load model
model = Finetuned_model(
    model_name, 
    device = device, 
    batch_size = BATCH_SIZE, 
    verbose = VERBOSE
    )

model_baseline = Finetuned_model(
    model_name, 
    device = device, 
    batch_size = BATCH_SIZE, 
    verbose = VERBOSE, 
    baseline=True
    )

# %% Proof of concept
model.predict(["he has a farm and is the taylor of fine dresses for all the ladys"], what = 5, get_dict = True)

# %% Predict 
# Get predictions. The following returns 
preds_w_lang = model.predict(df["occ1"].tolist(), lang = df["lang"].tolist(), what = 20)
preds_wo_lang = model.predict(df["occ1"].tolist(), what = 20)

embeddings_w_lang = model.forward_base(df["occ1"].tolist(), lang = df["lang"].tolist())
embeddings_wo_lang = model.forward_base(df["occ1"].tolist(), lang = df["lang"].tolist())

# Get baseline predictions for comparisons
embeddings_w_lang_base = model_baseline.forward_base(df["occ1"].tolist(), lang = df["lang"].tolist())
embeddings_wo_lang_base = model_baseline.forward_base(df["occ1"].tolist(), lang = df["lang"].tolist())

# %% Convert to data frame and save csv
df.to_csv(output_dir+"pred_data.csv") # Save background data 

preds_w_lang.to_csv(output_dir+"preds_w_lang.csv")
preds_wo_lang.to_csv(output_dir+"preds_wo_lang.csv")
pd.DataFrame(embeddings_w_lang).to_csv(output_dir+"embeddings_w_lang.csv")
pd.DataFrame(embeddings_wo_lang).to_csv(output_dir+"embeddings_wo_lang.csv")
pd.DataFrame(embeddings_w_lang_base).to_csv(output_dir+"embeddings_w_lang_base.csv")
pd.DataFrame(embeddings_wo_lang_base).to_csv(output_dir+"embeddings_wo_lang_base.csv")
