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
sys.path.append("../Model_scripts/")
from n103_Prediction_assets import Finetuned_model

import pandas as pd

# %% Load model
model = Finetuned_model(model_name, verbose = True)

# %% Load data
df = pd.read_csv("../Data/Application_data/Copenhagen Burial Records/transcribed_sources/CBP/CBP_20210309.csv")

df = df[["pa_id", "positions"]]
df = df.rename(columns = {"pa_id" : "RowID", "positions": "occ1"})
df = df[df['occ1'].notnull()]

df = df.sample(200, random_state = 20)
# %% Predict
pred05 = model.predict(df["occ1"].tolist(), lang = "da", what = "pred", threshold = 0.5)
fname05 = "../Data/Predictions/CopenhagenBurials_05.csv"
pred05.to_csv(fname05, sep = ";")

