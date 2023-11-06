# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 13:31:41 2023
https://medium.com/@keruchen/train-a-xlm-roberta-model-for-text-classification-on-pytorch-4ccf0b30f762
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
    elif(model_domain == "Multilingual"):
        fname = "../Data/Training_data"
    else:
        raise Exception("This is not implemented yet")
        
    return fname

def val_path(model_domain):
    if(model_domain == "DK_CENSUS"):
        fname = "../Data/Validation_data/DK_census_val.csv"
    elif(model_domain == "EN_MARR_CERT"):
        fname = "../Data/Validation_data/EN_marr_cert_val.csv" 
    elif(model_domain == "HSN_DATABASE"):
        fname = "../Data/Validation_data/HSN_database_val.csv"
    elif(model_domain == "Multilingual"):
        fname = "../Data/Validation_data"
    else:
        raise Exception("This is not implemented yet")
        
    return fname

#%% check_csv_column_consistency
def check_csv_column_consistency(folder_path):
    # Get a list of CSV files in the specified folder
    csv_files = [file for file in os.listdir(folder_path) if file.endswith(".csv")]

    if not csv_files:
        print("No CSV files found in the specified folder.")
        return

    # Read the first CSV file to get its column names
    first_file_path = os.path.join(folder_path, csv_files[0])
    first_df = pd.read_csv(first_file_path)
    first_columns = set(first_df.columns)

    # Initialize a variable to track whether all files have consistent columns
    consistent_columns = True

    # Check the columns of the remaining CSV files
    for file in csv_files[1:]:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path, nrows=10)
        if set(df.columns) != first_columns:
            consistent_columns = False
            print(f"Columns in '{file}' are not consistent with the first file.")

    # if consistent_columns:
    #     print("All CSV files in the folder have the same columns.")
    # else:
    #     print("Not all CSV files have the same columns.")
        
    return consistent_columns

# # Usage example
# check_csv_column_consistency(train_path(model_domain))

#%% Read_data
def read_data(model_domain, data_type = "Train", toyload = False):
    # breakpoint()
    if(model_domain == "Multilingual"):
        
        # Find correct path
        if data_type == "Train":
            fname = train_path(model_domain)
        elif data_type == "Validation":
            fname = val_path(model_domain)
        else:
           raise Exception("data_type not implemented yet")
        
        fnames = os.listdir(fname)
        
        # Check that all csv's have the same columns
        consistent_data = check_csv_column_consistency(fname)
        if not consistent_data:
            raise Exception("Problem in training data consistency. See above")
        
        # Initialize an empty dataframe to store the data
        combined_df = pd.DataFrame()
        # Loop through the file list and read each CSV file into the combined dataframe
        for file in fnames:
            if file.endswith(".csv"):  # Make sure the file is a CSV file
                file_path = os.path.join(fname, file)  # Replace with the actual path to your folder
                
                if toyload:
                    df = pd.read_csv(file_path, nrows = 100)
                else: 
                    df = pd.read_csv(file_path)
        
                combined_df = pd.concat([combined_df, df])
                print("\nRead "+file)
                
        df = combined_df
        
    else:
        # Find correct path
        if data_type == "Train":
            fname = train_path(model_domain)
        elif data_type == "Validation":
            fname = val_path(model_domain)
        else:
           raise Exception("data_type not implemented yet")
           
        if toyload:
            df = pd.read_csv(file_path, nrows = 100)
        else: 
            df = pd.read_csv(file_path)
    
    # Handle na strings
    df['occ1'] = df['occ1'].apply(lambda val: " " if pd.isna(val) else val)

    # Key
    key = pd.read_csv("../Data/Key.csv") # Load key and convert to dictionary
    key = key[1:]
    key = zip(key.code, key.hisco)
    key = list(key)
    key = dict(key)
    
    ### LONG TEXT TO GENERATE ERROR
    # df.occ1 = "kammerherre hab pai 5te aar waeret amkmand ower srmvancew hworwra hanhefter alledxnderdanigst ansoegningmer afgaaet med 300 rd pension ltamherrlwtil xhyrscech efter faderens doed"
    ###
    
    return df, key


# df, key = read_data(model_domain)
    
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

# %% Concat_string
# Makes one string with language and then occupational description
def Concat_string(occ1, lang):
    occ1 = str(occ1).strip("'[]'")
    # Implement random change to lang 'unknown' here:
    cat_sequence = "<s>"+lang+"</s></s>"+occ1+"</s>"
    
    return(cat_sequence)

#%% Dataset
class OCCDataset(Dataset):
    # Constructor Function 
    def __init__(
            self, 
            df, 
            tokenizer, 
            attacker, 
            max_len, 
            n_classes, 
            alt_prob = 0, 
            insert_words = False, 
            unk_lang_prob = 0.25 # Probability of changing lang to 'unknown'
            ):
        self.occ1 = df.occ1.tolist()
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.targets = labels_to_bin(df, n_classes)
        self.attacker = attacker
        self.alt_prob = alt_prob # Probability of text alteration in Attacker()
        self.insert_words = insert_words # Should random word insertation occur in Attacker()
        self.lang = df.lang.tolist()
        self.unk_lang_prob = unk_lang_prob
    
    # Length magic method
    def __len__(self):
        return len(self.occ1)
    
    # get item magic method
    def __getitem__(self, item):
        # breakpoint()
        occ1 = str(self.occ1[item])
        target = self.targets[item]
        lang = self.lang[item]
        
        # Implement Attack() here
        occ1 = self.attacker.attack(
            occ1, 
            alt_prob = self.alt_prob, 
            insert_words = self.insert_words
            )
        
        # breakpoint()
        # Change lanuage to 'unknown' = "unk" in some cases
        if(r.random()<self.unk_lang_prob):
            lang = "unk"
        
        occ1 = str(occ1).strip("'[]'")
        # Implement random change to lang 'unknown' here:
        cat_sequence = Concat_string(occ1, lang)
        
        # Encoded format to be returned 
        encoding = self.tokenizer.encode_plus(
            cat_sequence,
            add_special_tokens=True,
            padding = 'max_length',
            max_length = self.max_len,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',
            truncation = True
        )
        
        # Try to see if the following error can be caught here:
        # return torch.stack(batch, 0, out=out)
        #    RuntimeError: stack expects each tensor to be equal size, but got [63] at entry 0 and [50] at entry 1
        if encoding['input_ids'].shape[1] != self.max_len:
            # breakpoint()
            print(cat_sequence+" had shape: "+str(encoding['input_ids'].shape))
            print("This might cause an error")
        
        return {
            'occ1': cat_sequence,
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
        verbose = False,
        toyload = False,
        tokenizer = "No tokenizer" # If no tokenizer is provided one will be created
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

    # Load tokenizer (if non is provided)
    if tokenizer == "No tokenizer":
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
from n001_Model_assets import *
from n100_Attacker import *

# model_domain = "EN_MARR_CERT"

# df, key = read_data(model_domain)


# %% Load_val
# Simple loader for validation data

def Load_val(model_domain, sample_size, toyload = False):
    df, key = read_data(model_domain, data_type = "Validation", toyload = toyload)
    
    # Subset to smaller
    df = subset_to_smaller(df, sample_size=sample_size)
    
    df['concat_string0'] = [Concat_string(occ1, lang) for occ1, lang in zip(df['occ1'].tolist(), df['lang'].tolist())]
    df['concat_string1'] = [Concat_string(occ1, 'unk') for occ1 in df['occ1'].tolist()]
    
    # Make binary output matrix
    df_bin = labels_to_bin(df, max(df.code1)+1)
        
    return key, df, df_bin



