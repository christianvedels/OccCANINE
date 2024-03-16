# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 17:13:47 2024

@author: christian-vs
"""


import os
script_directory = os.path.dirname(os.path.abspath(__name__))
os.chdir(script_directory)

# %% Import necessary modules
from hisco import OccCANINE
import pandas as pd

# %% Load model
model = OccCANINE()

# %% Run
if __name__ == '__main__':
    # %% Load data
    df = pd.read_csv(
        "hisco/Data/TOYDATA.csv",
        nrows = 100
        )
    label_cols = ["hisco_1"]
    
    # Set lang
    df["lang"] = "en"  # English
    
    # %% Finetune HISCO
    model.finetune(
        df, 
        label_cols, 
        batch_size=32, 
        save_name = "Finetuned_toy_model",
        verbose_extra = True
        )
    
    # %% Finetuned model can be loaded
    model = OccCANINE("Finetuned/Finetuned_toy_model", hf = False)
    
    x = model.predict(["tailor of fine dresses"], lang = "en")
    print(x)
