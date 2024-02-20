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
from OccCANINE.n103_Prediction_assets import OccCANINE

# %% Load model
model = OccCANINE()

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

# %% Example 4 - 10000 (and more) are still very fast
import pandas as pd
df = pd.read_csv("Data/TOYDATA.csv")
model.verbose = True # Set updates to True
x = model.predict(
    df["occ1"],
    lang = "en",
    threshold = 0.22
    )

x["occ1"] = df["occ1"]
x.to_csv("Data/TOYDATA_wHISCO.csv")
