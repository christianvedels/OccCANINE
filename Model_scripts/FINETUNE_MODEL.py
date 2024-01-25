# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 17:13:47 2024

@author: christian-vs
"""


import os
script_directory = os.path.dirname(os.path.abspath(__name__))
os.chdir(script_directory)

# %% Import necessary modules
from n103_Prediction_assets import Finetuned_model
import pandas as pd

# %% Load model
model = Finetuned_model(
    name = "CANINE_Multilingual_CANINE_sample_size_10_lr_2e-05_batch_size_256",
    device = "cpu"
    )

# %% Fine tune
data_df = pd.read_csv("../Data/Training_data/SE_chalmers_train.csv", nrows = 5000)
label_cols = ["hisco_1", "hisco_2"]
model.finetune(data_df, label_cols, batch_size=16)



