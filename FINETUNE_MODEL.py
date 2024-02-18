# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 17:13:47 2024

@author: christian-vs
"""


import os
script_directory = os.path.dirname(os.path.abspath(__name__))
os.chdir(script_directory)

# %% Import necessary modules
from OccCANINE.n103_Prediction_assets import Finetuned_model
import pandas as pd

# %% Load model
model = Finetuned_model()

# %% Load data
df = pd.read_csv(
    "Data/TOYDATA_wHISCO.csv"
    )
label_cols = ["hisco_1", "hisco_2", "hisco_3", "hisco_4", "hisco_5"]

# Set lang
df["lang"] = "en"  # English

# %% Finetune
model.finetune(
    df, label_cols, batch_size=16, only_train_final_layer = True,
    save_name = "Finetune_toy_model"
    )

# %% Finetuned model can be loaded
model = Finetuned_model("Finetuned/Finetune_toy_model", hf = False)

x = model.predict(["tailor of fine dresses"], lang = "en")
print(x)
