# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 12:26:31 2024

@author: chris
"""

import os
script_directory = os.path.dirname(os.path.abspath(__name__))
os.chdir(script_directory)

# %% Libaries
import sys
sys.path.append("../OccCANINE/")
from n103_Prediction_assets import Finetuned_model

import pandas as pd

# %% Load model
model = Finetuned_model(verbose = True)

# %% Load data
df = pd.read_csv("../Data/Application_data/Copenhagen Burial Records/transcribed_sources/CBP/CBP_20210309.csv")

df = df[["pa_id", "positions"]]
df = df.rename(columns = {"pa_id" : "RowID", "positions": "occ1"})
df = df[df['occ1'].notnull()]

df = df.sample(200, random_state = 20)
# %% Predict
pred05 = model.predict(df["occ1"].tolist(), lang = "da", what = "pred", threshold = 0.34)
fname05 = "../Data/Predictions/CopenhagenBurials_05.csv"
pred05.to_csv(fname05, sep = ";")