# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 11:16:11 2024

@author: christian-vs

This module fine tunes the main model to each of set of training data

"""

import os
script_directory = os.path.dirname(os.path.abspath(__name__))
os.chdir(script_directory)

# %% Import necessary modules
from n103_Prediction_assets import Finetuned_model
import pandas as pd

# %% Load data
def load_data(fname, file, toyload = False):
    file_path = os.path.join(fname, file)  # Replace with the actual path to your folder
    
    if toyload:
        df = pd.read_csv(file_path, nrows = 500)
    else: 
        df = pd.read_csv(file_path)
        
    return df

# %% Load model
def load_model():
    model = Finetuned_model(
        name = "CANINE_Multilingual_CANINE_sample_size_10_lr_2e-05_batch_size_256"
        )
    return model

# %% Train
def train(model, data_df, name):
    label_cols = ["hisco_1", "hisco_2", "hisco_3", "hisco_4", "hisco_5"]
    model.finetune(
        data_df, 
        label_cols, 
        batch_size=16, 
        only_train_final_layer = True,
        epochs = 2,
        save_name = name+"_finetuneCANINE"
        )

# %% Loop
fname = "../Data/Training_data"
fnames = os.listdir(fname)

for file in fnames:
    if file.endswith(".csv"):  # Make sure the file is a CSV file
        print(f"----> Finetuning on {file}")
        df = load_data(fname, file, toyload=True)
        model = load_model()
        train(model, df, name = file[:-4])
