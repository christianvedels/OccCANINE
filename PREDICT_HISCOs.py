# -*- coding: utf-8 -*-
"""
Created on 2024-01-15

@author: christian-vs

Prediction for applications

SETUP:
    - See readme2.md
"""

import os
script_directory = os.path.dirname(os.path.abspath(__name__))
os.chdir(script_directory)

# %% Import necessary modules
from OccCANINE.n103_Prediction_assets import Finetuned_model

# %% Load model
model = Finetuned_model()

# %% Example 1
model.predict(
    ["tailor of the finest suits"], 
    lang = "en", 
    get_dict = True, 
    threshold = 0.22 # Best F1 for English
    )

# %% Example 2
model.predict(
    ["the train's fireman"], 
    lang = "en", 
    get_dict = True, 
    threshold = 0.22 # Best F1 for English
    )

# %% Example 3
model.predict(
    ["nurse at the local hospital"],
    lang = "en", 
    get_dict = True, 
    threshold = 0.22 # Best F1 for English
    )

# %% Example 4 - 1000 (and more) are still very fast
import pandas as pd
df = pd.read_csv("Data/TOYDATA.csv")
print(f"Producing HISCO codes for {df.shape[0]} observations")
print(f"Estimated human hours saved: {df.shape[0]*10/60/60} hours")
model.verbose = True # Set updates to True
x = model.predict(
    df["occ1"],
    lang = "en",
    threshold = 0.22
    )

x.to_csv("Data/TOYDATA_wHISCO.csv")
