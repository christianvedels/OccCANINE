# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 15:37:19 2023

@author: christian-vs
"""



# %% Parameters
 
#################################
chars = "occ1"
hisco_level = 5
nation = "DK"
#################################

#%% Libraries
import os
import numpy as np
from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Input
import tensorflow as tf
import pandas as pd
import random as r
import matplotlib.pyplot as plt
import json
from keras_preprocessing.text import tokenizer_from_json
from keras_preprocessing.sequence import pad_sequences
import random
    
#%% Load data
if os.environ['COMPUTERNAME'] == 'SAM-126260':
    os.chdir('D:/Dropbox/PhD/HISCO')
else:
    os.chdir('C:/Users/chris/Dropbox/PhD/HISCO')
print(os.getcwd())

# fname = "Data/HISCO_all"+str(hisco_level)+".csv"
# fname = "Data_Danish_migrants/Clean_occ_data.csv"
fname = "Data_human_cap_nordics/Occupations_dk.csv"
df = pd.read_csv(fname, encoding = "UTF-8", sep = ";")
# df0 = df.copy()


# fname = "Data/Link lives data/Occupation_strings.csv"
# df = pd.read_csv(fname, encoding = "UTF-8")
# df0 = df.copy()

# fname = "Data/Toydata"+str(hisco_level)+".csv"
# df = pd.read_csv(fname , encoding = "UTF-8")

# Subset to only train
# df = df[df["validation"]==1]
# df = df[df.no_occ != 1]
strings = df[chars]

# Load keys
key = pd.read_csv("Data/key5.csv")

# %% Eval metrics
from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + 
    K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# %% Load model
if nation == "DK":
    fname = "Models/Mod_char_aug0.2_unique_False_smplsize_10_lvl5"
    # fname = "Models/Mod_char_aug0_unique_False_smplsize_10_lvl5"

if nation == "UK":
    fname = "Models/Mod_charUK_aug0.2_unique_False_smplsize_10_lvl5"

mod_char = tf.keras.models.load_model(fname, custom_objects={'f1_m':f1_m, 'precision_m': precision_m, 'recall_m': recall_m})


#%% Function to convert letters if any
# Written by chatGPT
def clean_string(text):
    
    # Make all strings lowercase
    if isinstance(text, pd.Series):
        text = text.str.lower()
    else:
        text = text.lower()
    
    # Replace scandinavian letters
    scandinavian_letters = {'æ': 'ae', 'ø': 'oe', 'å': 'aa', 'ä': 'ae', 'ö': 'oe'}
    if isinstance(text, str):
        text = [text]
    result = []
    for item in text:
        for key, value in scandinavian_letters.items():
            item = item.replace(key, value)
        result.append(item)
    if len(result) == 1:
        return result[0]
    else:
        return result
    
#%% Return0.5
def return05(pred, x_batch):
    which_codes = [np.where(i > 0.5) for i in pred]
    which_codes = [list(i[0][:]+3) for i in which_codes]
    hisco = [key.hisco[i].tolist() for i in which_codes]
    desc = [key.en_hisco_text[i].tolist() for i in which_codes]
    hisco = [[-1] if len(i)==0 else i for i in hisco]
    desc = [['Missing, no title'] if len(i)==0 else i for i in desc]
    result = pd.DataFrame(zip(x_batch, hisco, desc))
    return result

def returnTop5(pred, x_batch):
    which_codes = np.argsort(pred, axis=1)[:, ::-1][:, :5] + 3
    hisco = [key.hisco[i].tolist() for i in which_codes]
    desc = [key.en_hisco_text[i].tolist() for i in which_codes]
    hisco = [[-1] if len(i)==0 else i for i in hisco]
    desc = [['Missing, no title'] if len(i)==0 else i for i in desc]
    prob5 = [[pred[row, col] for col in indices] for row, indices in enumerate(which_codes-3)]
    result = pd.DataFrame(zip(x_batch, hisco, desc, prob5))
    return result

#%% Make prediction and give HISCO code
def pred_it(mod, X, rowID, name, thresh = True, batch_size = 1024, return_HISCO = True, return_Top5 = True):
    X = clean_string(X)
    n = len(X)
    
    # Check if the Data directory exists and create it if it doesn't
    if not os.path.exists("Tmp1"):
        os.makedirs("Tmp1")
        
    if not os.path.exists("Tmp2"):
        os.makedirs("Tmp2")
        
    # If thresh not defined:
    if thresh:
        # Get output dimensions
        K = mod.predict(["dummy entry"]).shape[1]
        thresh = np.full(K, 0.5)
    
    # Loop over X in batches of size batch_size
    for i in range(0, n, batch_size):
        print("\n"+str(i)+" of", str(n))
        
        x_batch = X[i:i+batch_size]
        
        # Make predictions on current batch
        pred = mod.predict(x=x_batch, batch_size=64)
        result = pd.DataFrame(pred)
        
        if return_HISCO:
            result = return05(pred, x_batch)
            
        if return_Top5:
            result_top5 = returnTop5(pred, x_batch)
                
        # Save batch result to CSV file
        save_fname = f"Tmp1/{i//batch_size+1}.csv"
        result.to_csv(save_fname)
        
        save_fname = f"Tmp2/{i//batch_size+1}.csv"
        result_top5.to_csv(save_fname)
        
    # Concatenate all CSV files into a single DataFrame
    csv_files = [f"Tmp1/{i+1}.csv" for i in range(n//batch_size + 1)]
    concat_df = pd.concat([pd.read_csv(f) for f in csv_files])
    concat_df.insert(0, "RowID", rowID.tolist())
    
    csv_files = [f"Tmp2/{i+1}.csv" for i in range(n//batch_size + 1)]
    concat_df_top5 = pd.concat([pd.read_csv(f) for f in csv_files])
    concat_df_top5.insert(0, "RowID", rowID.tolist())
    
    # Save concatenated DataFrame to a single CSV file
    concat_fname = f"{name}.csv"
    concat_df.to_csv(concat_fname, index=False, sep = ";")
    concat_fname = f"{name}_top5.csv"
    concat_df_top5.to_csv(concat_fname, index=False, sep = ";")
    
    # Saving sample
    # Saving sample
    random.seed(20)  # Set the random seed to 20
    sample_data = concat_df.sample(n=100)  # Sample 100 random observations from concat_df
    sample_fname = f"{name}_sample100.csv"  # Define the file name for the sample CSV
    sample_data.to_csv(sample_fname, index=False, sep = ";")  # Save the sampled data to the CSV

    # Delete temporary CSV files
    for f in csv_files:
        os.remove(f)
        
    return(concat_df)

#%% Toy data
# Faraoe Island occupations
# strings = [
#     'købmand',
#     'handelsfuldmægtig',
#     'fisker',
#     'købmand',
#     'købmand',
#     'fisker',
#     'købmand',
#     'fisker',
#     'postekspedient',
#     'købmand',
#     'fisker',
#     'købmand',
#     'købmand',
#     'landmand',
#     'laerer',
#     'postmester',
#     'købmand',
#     'trykkeribestyrer',
#     'købmand',
#     'købmand',
#     'skipper',
#     'lods.',
#     'uhrmager',
#     'lærer',
#     ]

#%% Run prediction

res1 = pred_it(
    mod = mod_char,
    X = strings,
    name = "Data_human_cap_nordics/Predicted_HISCO_codes",
    rowID = df.RowID
    )

# res1.to_csv("Tmp.csv")

    