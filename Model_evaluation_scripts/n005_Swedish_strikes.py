# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 12:26:31 2024

@author: chris
"""

import os
script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)
print(os.getcwd())

# %% Libaries
import sys
module_path = os.path.join(script_directory, "..", "OccCANINE")
sys.path.append(module_path)
from n103_Prediction_assets import OccCANINE

import pandas as pd

# %% Load model
model = OccCANINE(verbose = True, device="cpu")

# %% Load data
df0 = pd.read_excel("../Data/Application_data/Swedish_Strikes/National_Sweden_1859-1902.xlsx")
df = df0.sample(200, random_state = 20)
df = df[["profession", "hisco"]]
# %% Predict
pred = model.predict(df["profession"].tolist(), lang = "se", what = "pred", threshold = 0.45)
fname = "../Data/Predictions/SwedishStrikes.csv"

# Reset index before concatenation
df.reset_index(drop=True, inplace=True)
pred.reset_index(drop=True, inplace=True)

df = pd.concat([pred, df], axis = 1)

# Add descriptions from key
key = pd.read_csv("../Data/Key.csv")
key['hisco'] = pd.to_numeric(key['hisco'], errors='coerce').astype('float64')
key = key[["en_hisco_text", "hisco"]]
df = df.merge(key, on='hisco', how='left')

# %% Save
df.to_csv(fname, sep = ";", encoding = "ISO-8859-1")

# %% Finetune it
df0 = df0.rename(columns={"profession": "occ1", "hisco": "hisco_1"})
df0["lang"] = "se"
model1 = model.finetune(
    df0,
    label_cols = ["hisco_1"],
    epochs = 20,
    batch_size = 64,
    verbose_extra=True
    )
