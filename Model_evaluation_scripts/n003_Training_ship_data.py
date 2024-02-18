# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 15:09:49 2024

@author: chris
"""

import os
script_directory = os.path.dirname(os.path.abspath(__name__))
os.chdir(script_directory)

# %% Libaries
import sys
module_path = os.path.join(script_directory, "..", "OccCANINE")
sys.path.append(module_path)
from n103_Prediction_assets import Finetuned_model

import pandas as pd

# %% Load model
model = Finetuned_model(verbose = True)

# %% Load data
df_early = pd.read_stata("../Data/Application_data/Training_ship_data/indy_early.dta")
df_late = pd.read_stata("../Data/Application_data/Training_ship_data/indy_late.dta")
df = pd.concat([df_early, df_late])

df_f = df[["id", "f_occ", "f_occ_hisco"]]
df_m = df[["id", "m_occ", "m_occ_hisco"]]
df_f = df_f.rename(columns = {"f_occ" : "occ1", "f_occ_hisco":"hisco"})
df_m = df_m.rename(columns = {"m_occ" : "occ1", "m_occ_hisco":"hisco"})

df0 = pd.concat([df_f, df_m])
df0 = df0[df0['occ1']!=""]

df = df0.sample(200, random_state = 20)

# Add descriptions from key
key = pd.read_csv("../Data/Key.csv")
key['hisco'] = pd.to_numeric(key['hisco'], errors='coerce').astype('float64')
key = key[["en_hisco_text", "hisco"]]
df = df.merge(key, on='hisco', how='left')

# %% Predict
pred = model.predict(df["occ1"].tolist(), lang = "en", what = "pred", threshold = 0.47)
fname = "../Data/Predictions/TrainingShip.csv"

df = pd.concat([pred, df], axis = 1)

# %% Save
df.to_csv(fname, sep = ";")

