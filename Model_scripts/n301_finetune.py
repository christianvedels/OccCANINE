# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 11:16:11 2024

@author: christian-vs

This module fine tunes the main model to each of set of training data

"""

import os
script_directory = os.path.dirname(os.path.abspath(__name__))
os.chdir(script_directory)

# %% Parameters (iterates in loop)
nobs_values = [1000, 2000, 10000, 100000]
train_final_layer_values = [False, True]

for NOBS in nobs_values:
    for ONLY_TRAIN_FINAL_LAYER in train_final_layer_values:
        print(f"NOBS: {NOBS}")
        print(f"ONLY_TRAIN_FINAL_LAYER: {ONLY_TRAIN_FINAL_LAYER}")

        # %% Import necessary modules
        from n103_Prediction_assets import Finetuned_model
        import pandas as pd
        import logging
        
        # %% Load data
        def load_data(fname, file, nrows, toyload = False):
            file_path = os.path.join(fname, file)  # Replace with the actual path to your folder
            
            if toyload:
                df = pd.read_csv(file_path, nrows = nrows)
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
                batch_size=128, 
                only_train_final_layer = ONLY_TRAIN_FINAL_LAYER,
                epochs = 50,
                save_name = f"{name}_{NOBS}_trainall{not ONLY_TRAIN_FINAL_LAYER}_finetuneCANINE",
                verbose_extra = True,
                test_fraction = 0.2
                )
        
        # %% Loop
        fname = "../Data/Training_data"
        fnames = ['EN_ship_data_train.csv', 'HSN_database_train.csv','SE_swedpop_train.csv', 'DK_census_train.csv']
        
        for file in fnames:
            if file.endswith(".csv"):  # Make sure the file is a CSV file
                print(f"----> Finetuning on {file}")
                df = load_data(fname, file, nrows = NOBS, toyload=True)
                model = load_model()
                train(model, df, name = file[:-4])
