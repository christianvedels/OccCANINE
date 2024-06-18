# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 15:09:49 2024

@author: chris
"""

# %% Libaries
from hisco import OccCANINE
import pandas as pd
import os

# %% Params
# Choose which GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# %% Load model
model = OccCANINE(verbose = True)

if __name__ == '__main__':
    
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
    
