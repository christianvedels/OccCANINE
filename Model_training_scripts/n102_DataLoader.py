# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 13:31:41 2023

@author: chris
"""
import os
script_directory = os.path.dirname(os.path.abspath(__name__))
os.chdir(script_directory)

#%% Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random as r
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch

#%% Returns training data path
def train_path(model_domain):
    if(model_domain == "DK_CENSUS"):
        fname = "../Data/Training_data/DK_census_train.csv"
    elif(model_domain == "EN_MARR_CERT"):
        fname = "../Data/Training_data/EN_marr_cert_train.csv" 
    elif(model_domain == "HSN_DATABASE"):
        fname = "../Data/Training_data/HSN_database_train.csv"
    else:
        raise Exception("This is not implemented yet")
        
    return fname

#%% Read_data
def read_data(model_domain):
    # breakpoint()
    fname = train_path(model_domain)
    
    df = pd.read_csv(fname, encoding = "UTF-8")

    # Handle na strings
    df['occ1'] = df['occ1'].apply(lambda val: " " if pd.isna(val) else val)

    # Key
    key = pd.read_csv("../Data/Key.csv") # Load key and convert to dictionary
    key = key[1:]
    key = zip(key.code, key.hisco)
    key = list(key)
    key = dict(key)
    
    return df, key
    
#%% Resample()
def resample(
        df,
        downsample_top1 = True,
        upsample_below = 0,
        verbose = False        
        ):
    # Downsapmling "no occupation"
    # Find the largest category
    category_counts = df['code1'].value_counts()

    if verbose:
        # Make plot
        sns.distplot(category_counts.tolist())
        plt.xlabel('Labels count (log scale)')
        plt.xscale('log')
        plt.show()

    if downsample_top1:
        # Downsample
        largest_category = category_counts.index[0]
        second_largest_size = next_largest_cat = category_counts.tolist()[1]

        # Split df into df with the largest category and df with other categories
        df_largest = df[df.code1 == largest_category]
        df_other = df[df.code1 != largest_category]

        # Downsample to size of next largest cat if it is the largest
        df_largest = df_largest.sample(second_largest_size, random_state=20)

        # Merge 'df_noocc' and 'df_occ' into 'df' and shuffle data
        df = pd.concat([df_largest, df_other], ignore_index=True)
        df = df.sample(frac=1, random_state=20)  # Shuffle the rows

        # Print new counts
        if verbose:
            category_counts = df['code1'].value_counts() 
            print(category_counts)
        
    if upsample_below>0:
        # Upsampling the remaining data
        # Labels with less than 'upsample_below' have an 'upsample_below' observations
        # added to the data

        # Initialize an empty DataFrame to store upsampled samples
        upsampled_df = pd.DataFrame()

        # Loop through unique classes (excluding the 'no occupation' class)
        for class_label in df['code1'].unique():
            class_samples = df[df['code1'] == class_label]
            if(class_samples.shape[0]==0):
                continue
            if(class_samples.shape[0]<upsample_below):
                if verbose:
                    print(f"Upsampling: {class_samples.shape[0]} --> {upsample_below+class_samples.shape[0]}")
                oversampled_samples = class_samples.sample(upsample_below, replace=True, random_state=20)
                upsampled_df = pd.concat([upsampled_df, oversampled_samples], ignore_index=True)

        # Combine upsampled data with 'no occupation' downsampled data
        df = pd.concat([df, upsampled_df], ignore_index=True)
        df = df.sample(frac=1, random_state=20)  # Shuffle the rows again
        
        if verbose:
            # Print new counts after upsampling
            category_counts = df['code1'].value_counts() 
            print(category_counts)

            # Make plot
            sns.distplot(category_counts.tolist())
            plt.xlabel('Labels count (log scale)')
            plt.xscale('log')
            plt.show()
            
    # Return result    
    return df
        
#%% Downsample
# Subset to smaller
def subset_to_smaller(df, sample_size):
    if(10**sample_size < df.shape[0]):
        r.seed(20)
        df = df.sample(10**sample_size, random_state=20) 
        
    return df

# %% Labels to bin function
# Returns binary array
def labels_to_bin(df, max_value):
    df_codes = df[["code1", "code2", "code3", "code4", "code5"]]
    # Binarize
    labels_list = df_codes.values.tolist()
    # == Build outcome matrix ==
    # Construct the NxK matrix
    N = len(labels_list)
    K = int(max_value)
    labels = np.zeros((N, K), dtype=int)

    for i, row in enumerate(labels_list):
        for value in row:
            if not np.isnan(value):
                labels[i, int(value)] = 1


    # The columns 0-2 contains all those labelled as having 'no occupation'.
    # But since any individuals is encoded with up 5 occupations, and any one o
    # these can be 'no occupation' it occurs erroneously with high frequency.
    # Fix: Check if any other occupation is positve.
    # Iterate through each row of the array
    for row in labels:
        # Check if the value in the first column is '1'
        if row[2] == 1 | row[1] == 1 | row[0] == 1:
            # Check if there are any positive values in the remaining row (excluding the first column)
            if np.any(row[3:]>0):
                # Set the first column to '0' if positive values are found
                row[0] = 0

    labels = labels.astype(float)
     
    # Convert each row of the array to a 1D list
    # labels = [row.tolist() for row in labels]
     
    return(labels)

# %% Reference loss
def referenceLoss(df):
    # Step 1: Calculate the probabilities for each class (frequency of occurrence)
    labels = labels_to_bin(df, max(df.code1)+1)
    probs = np.mean(labels, axis=0)

    # Step 2: Calculate the binary cross entropy for each class with epsilon to avoid divide by zero and log of zero
    epsilon = 1e-9
    probs = np.clip(probs, epsilon, 1 - epsilon)  # Clip probabilities to be in range [epsilon, 1-epsilon]

    # BCE formula: -y * log(p) - (1-y) * log(1-p)
    bce_per_class = -(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))

    # Step 3: Sum up the binary cross entropy values for all classes
    reference_bce = np.mean(bce_per_class)

    return reference_bce 


# %% ReadData
def ReadData(
        model_domain, 
        downsample_top1 = True,
        upsample_below = 1000,
        sample_size = 4,
        verbose = False
        ):
    df, key = read_data(model_domain = model_domain)
    df = resample(df, downsample_top1=downsample_top1, upsample_below=upsample_below, verbose=verbose)
    df = subset_to_smaller(df, sample_size=sample_size)
    
    return df, key

# %% Train test val split
def TrainTestVal(df, verbose = False):
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=20)
    df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=20)
    if verbose:
        print(f"Train {df_train.shape[0]} / Val {df_val.shape[0]} / Test {df_test.shape[0]}")
        
    return df_train, df_val, df_test

#%% Dataset
class OCCDataset(Dataset):
    # Constructor Function 
    def __init__(self, df, tokenizer, attacker, max_len, n_classes, alt_prob = 0, insert_words = False):
        self.occ1 = df.occ1.tolist()
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.targets = labels_to_bin(df, n_classes)
        self.attacker = attacker
        self.alt_prob = alt_prob # Probability of text alteration in Attacker()
        self.insert_words = insert_words # Should random word insertation occur in Attacker()
    
    # Length magic method
    def __len__(self):
        return len(self.occ1)
    
    # get item magic method
    def __getitem__(self, item):
        # breakpoint()
        occ1 = str(self.occ1[item])
        target = self.targets[item]
        
        # Implement Attack() here
        occ1 = self.attacker.attack(
            occ1, 
            alt_prob = self.alt_prob, 
            insert_words = self.insert_words
            )
        
        # Encoded format to be returned 
        encoding = self.tokenizer.encode_plus(
            occ1,
            add_special_tokens=True,
            padding = 'max_length',
            max_length = self.max_len,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'occ1': occ1,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }

#%% Function to return datasets    
def datasets(df_train, df_val, df_test, 
             tokenizer,
             attacker,
             max_len,
             n_classes,
             alt_prob, 
             insert_words):
    
    # Only 'ds_train_attack' has any attack probability
    ds_train = OCCDataset(
        df=df_train,
        tokenizer=tokenizer,
        attacker=attacker,
        max_len=max_len,
        n_classes=n_classes,
        alt_prob = 0, 
        insert_words = False
    )
    ds_train_attack = OCCDataset(
        df=df_train,
        tokenizer=tokenizer,
        attacker=attacker,
        max_len=max_len,
        n_classes=n_classes,
        alt_prob = alt_prob, 
        insert_words =insert_words 
    )
    ds_val = OCCDataset(
        df=df_val,
        tokenizer=tokenizer,
        attacker=attacker,
        max_len=max_len,
        n_classes=n_classes,
        alt_prob = 0, 
        insert_words = False
    )    
    ds_test = OCCDataset(
        df=df_test,
        tokenizer=tokenizer,
        attacker=attacker,
        max_len=max_len,
        n_classes=n_classes,
        alt_prob = 0, 
        insert_words = False
    )
    
    return ds_train, ds_train_attack, ds_val, ds_test

#%%
def create_data_loader(ds_train, ds_train_attack, ds_val, ds_test, batch_size):
    
    data_loader_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        num_workers=0
    )
    
    data_loader_train_attack = DataLoader(
        ds_train_attack,
        batch_size=batch_size,
        num_workers=0
    )
    
    data_loader_val = DataLoader(
        ds_val,
        batch_size=batch_size,
        num_workers=0
    )
    
    data_loader_test = DataLoader(
        ds_test,
        batch_size=batch_size,
        num_workers=0
    )
    
    return data_loader_train, data_loader_train_attack, data_loader_val, data_loader_test

# %% Load data
def Load_data(
        model_domain = "DK_CENSUS",
        downsample_top1 = True,
        upsample_below = 1000,
        sample_size = 4,
        max_len = 50,
        alt_prob = 0.1,
        insert_words = True,
        batch_size = 16,
        verbose = False
        ):
    
    # Load data
    df, key = ReadData(
        model_domain, 
        downsample_top1 = downsample_top1,
        upsample_below = upsample_below,
        sample_size = sample_size,
        verbose = verbose
        )
    df_train, df_val, df_test = TrainTestVal(df, verbose=verbose)

    # Load tokenizer
    tokenizer = load_tokenizer(model_domain)
    tokenizer = update_tokenizer(tokenizer, df)

    # Calculate number of classes
    N_CLASSES = len(key)
    
    # Define attakcer instance
    attacker = AttackerClass(df)
    
    # Calculate reference loss
    reference_loss = referenceLoss(df)

    # Datsets
    ds_train, ds_train_attack, ds_val, ds_test = datasets(
        df_train, df_val, df_test,
        tokenizer=tokenizer,
        attacker=attacker,
        max_len=max_len,
        n_classes=N_CLASSES,
        alt_prob = alt_prob, 
        insert_words = insert_words
        )
    
    # Data loaders
    data_loader_train, data_loader_train_attack, data_loader_val, data_loader_test = create_data_loader(
        ds_train, ds_train_attack, ds_val, ds_test,
        batch_size = batch_size
        )
    
    return {
        'data_loader_train': data_loader_train,
        'data_loader_train_attack': data_loader_train_attack,
        'data_loader_val': data_loader_val,
        'data_loader_test': data_loader_test,
        'tokenizer': tokenizer,
        'N_CLASSES': N_CLASSES,
        'key': key,
        'reference_loss': reference_loss
    }

#%%
from n001_BERT_models import *
from n100_Attacker import *

# model_domain = "EN_MARR_CERT"

# df, key = read_data(model_domain)

