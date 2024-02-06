# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 15:09:49 2024

@author: chris
"""

import os
script_directory = os.path.dirname(os.path.abspath(__name__))
os.chdir(script_directory)

# %% Params
# Choose which GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Name of the finetuned model to use (must be located in "Trained_models") 
model_name = "CANINE_Multilingual_CANINE_sample_size_10_lr_2e-05_batch_size_256"

# %% Libaries
import sys
sys.path.append("..\\Model_scripts\\")
from n103_Prediction_assets import Finetuned_model

import pandas as pd

# %% Load model
model = Finetuned_model(model_name, verbose = True, device = "cpu")

# %% Load data
df_early = pd.read_stata("../Data/Application_data/Training_ship_data/indy_early.dta")
df_late = pd.read_stata("../Data/Application_data/Training_ship_data/indy_late.dta")
df = pd.concat([df_early, df_late])

df_f = df[["id", "f_occ"]]
df_m = df[["id", "m_occ"]]
df_f = df_f.rename(columns = {"f_occ" : "occ1"})
df_m = df_m.rename(columns = {"m_occ" : "occ1"})

df = pd.concat([df_f, df_m])

df = df.sample(200, random_state = 20)
# %% Predict
pred = model.predict(df["occ1"].tolist(), lang = "en", what = "pred", threshold = 0.5)
fname = "../Data/Predictions/TrainingShip.csv"
pred.to_csv(fname, sep = ";")