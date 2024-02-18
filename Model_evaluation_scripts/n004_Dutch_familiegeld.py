# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 10:58:14 2024

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
# Define a function to read the text file and extract information
def read_occupations(file_path):
    occupations = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()  # Remove leading/trailing whitespace
            if line:  # Check if the line is not empty
                # breakpoint()
                parts = line.split(",")  # Split the line at the comma
                # For demonstration, assuming each line is just the occupation
                if len(parts) > 1:
                    occupation_part = parts[-1].strip()  # Strip leading/trailing whitespace
                    occupation_parts = occupation_part.split()  # Split on whitespace
                    if len(occupation_parts) > 1:
                        occupation = occupation_parts[-2]  # The second last part is assumed to be the occ. 
                        quantity = occupation_parts[-1]  # The last part after cleaning, assumed to be the quantity here
                        found_occ_description = 1
                    else:
                        occupation = " ".join(occupation_parts) 
                        found_occ_description = 0
                else:
                    occupation = line
                    quantity = "NA"
                    found_occ_description = 0
                # Placeholder for HISCO identification, you would replace this with the actual model prediction
                occupations.append({"original_text": line,"occupation": occupation, "tax": "x"+quantity, "found_occ_description": found_occ_description})
    
    return pd.DataFrame(occupations)

file_path = '../Data/Application_data/Familiegeld/Familiegeld.txt'
df_occupations = read_occupations(file_path)
df_occupations = df_occupations[df_occupations['found_occ_description'] == 1]
df_occupations = df_occupations.drop('found_occ_description', axis = 1)

df_sample = df_occupations.sample(200, random_state = 20)

# %% Predict
# All
pred_all = model.predict(df_occupations["occupation"].tolist(), lang = "nl", what = "pred", threshold = 0.37)
fname_all = "../Data/Predictions/Familiegeld_all.csv"

# Sample
pred_samples = model.predict(df_sample["occupation"].tolist(), lang = "nl", what = "pred", threshold = 0.37)
fname_sample = "../Data/Predictions/Familiegeld_sample.csv"

# Reset index before concatenation
df_occupations.reset_index(drop=True, inplace=True)
df_sample.reset_index(drop=True, inplace=True)

# Assuming pred_all and pred_samples are pandas DataFrames and have a matching number of rows as their corresponding DataFrames
pred_all.reset_index(drop=True, inplace=True)
pred_samples.reset_index(drop=True, inplace=True)

# Concatenate with ignore_index=True
df_all = pd.concat([pred_all, df_occupations], axis=1)
df_sample = pd.concat([pred_samples, df_sample], axis=1)

# %% Save
df_all.to_csv(fname_all, sep = ";", index = False)
df_sample.to_csv(fname_sample, sep = ";", index = False)