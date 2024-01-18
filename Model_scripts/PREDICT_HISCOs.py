# -*- coding: utf-8 -*-
"""
Created on 2024-01-15

@author: christian-vs

Prediction for applications
"""

import os
script_directory = os.path.dirname(os.path.abspath(__name__))
os.chdir(script_directory)

# %% Import necessary modules
import torch
from n103_Prediction_assets import Finetuned_model

#%% Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% Load model
model = Finetuned_model(
    model_name = "CANINE_Multilingual_CANINE_sample_size_10_lr_2e-05_batch_size_256", 
    device = device, 
    batch_size = 256, 
    verbose = True
    )

# %% Use the model
model.predict(
    ["taylor of fne dresses"],
    what = "pred"
    )