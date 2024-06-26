# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 13:31:41 2023
https://medium.com/@keruchen/train-a-xlm-roberta-model-for-text-classification-on-pytorch-4ccf0b30f762
@author: chris
"""


import io
import os
import random

from functools import partial
from typing import Callable

import torch

from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from transformers import CanineTokenizer

import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

# Custom modules
from .datasets import DATASETS
from .model_assets import load_tokenizer, update_tokenizer
from .attacker import AttackerClass
from .formatter import BlockyHISCOFormatter


# Returns training data path
def train_path(model_domain): # FIXME move to datasets.py and avoid hardcoded paths
    if model_domain == "DK_CENSUS":
        fname = "../Data/Training_data/DK_census_train.csv"
    elif model_domain == "EN_MARR_CERT":
        fname = "../Data/Training_data/EN_marr_cert_train.csv"
    elif model_domain == "HSN_DATABASE":
        fname = "../Data/Training_data/HSN_database_train.csv"
    elif model_domain == "Multilingual":
        fname = "../Data/Training_data"
    elif model_domain == "Multilingual_CANINE":
        fname = "../Data/Training_data"
    else:
        raise NotImplementedError("This is not implemented yet")

    return fname

def val_path(model_domain): # FIXME move to datasets.py and avoid hardcoded paths
    if model_domain == "DK_CENSUS":
        fname = "../Data/Validation_data/DK_census_val.csv"
    elif model_domain == "EN_MARR_CERT":
        fname = "../Data/Validation_data/EN_marr_cert_val.csv"
    elif model_domain == "HSN_DATABASE":
        fname = "../Data/Validation_data/HSN_database_val.csv"
    elif model_domain == "Multilingual":
        fname = "../Data/Validation_data"
    elif model_domain == "Multilingual_CANINE":
        fname = "../Data/Validation_data"
    else:
        raise NotImplementedError("This is not implemented yet")

    return fname

#check_csv_column_consistency
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

    return consistent_columns


def read_data(model_domain, data_type = "Train", toyload = False, verbose = True):
    if model_domain == "Multilingual" or model_domain == "Multilingual_CANINE":

        # Find correct path
        if data_type == "Train":
            fname = train_path(model_domain)
        elif data_type == "Validation":
            fname = val_path(model_domain)
        else:
            raise NotImplementedError("data_type not implemented yet")

        fnames = os.listdir(fname)

        # Check that all csv's have the same columns
        consistent_data = check_csv_column_consistency(fname)
        if not consistent_data:
            raise NotImplementedError("Problem in training data consistency. See above")

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

                n_df = df.shape[0]

                combined_df = pd.concat([combined_df, df])
                if verbose:
                    print(f"\nRead {file} (N = {n_df})")

        df = combined_df

    else:
        # Find correct path
        if data_type == "Train":
            fname = train_path(model_domain)
        elif data_type == "Validation":
            fname = val_path(model_domain)
        else:
            raise NotImplementedError("data_type not implemented yet")

        if toyload:
            df = pd.read_csv(fname, nrows = 100)
        else:
            df = pd.read_csv(fname)

    # Handle na strings
    df['occ1'] = df['occ1'].apply(lambda val: " " if pd.isna(val) else val)

    # Key
    key = DATASETS['keys']() # Load key and convert to dictionary
    key = key[1:]
    key = zip(key.code, key.hisco)
    key = list(key)
    key = dict(key)

    return df, key

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
        sns.distplot(category_counts.tolist()) # FIXME sns.distplot is deprecated
        plt.xlabel('Labels count (log scale)')
        plt.xscale('log')
        plt.show()

    if downsample_top1:
        # Downsample
        largest_category = category_counts.index[0]
        second_largest_size = category_counts.tolist()[1]

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

    if upsample_below > 0:
        # Upsampling the remaining data
        # Labels with less than 'upsample_below' have an 'upsample_below' observations
        # added to the data

        # Initialize an empty DataFrame to store upsampled samples
        upsampled_df = pd.DataFrame()

        # Loop through unique classes (excluding the 'no occupation' class)
        for class_label in df['code1'].unique():
            class_samples = df[df['code1'] == class_label]
            if class_samples.shape[0]==0:
                continue
            if class_samples.shape[0]<upsample_below:
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

    return df


def subset_to_smaller(df, sample_size):
    ''' Downsample
    '''
    if 10 ** sample_size < df.shape[0]:
        random.seed(20)
        df = df.sample(10**sample_size, random_state=20)

    return df


def labels_to_bin(df: pd.DataFrame, max_value: int):
    ''' Returns binary array
    '''
    df_codes = df[["code1", "code2", "code3", "code4", "code5"]]

    if len(df) == 1: # Handle single row efficiently
        # Directly work with the single row's values
        row_values = df_codes.iloc[0].values
        labels = np.zeros(max_value, dtype=int)
        for value in row_values:
            if not np.isnan(value):
                labels[int(value)] = 1

        # 'No occupation' correction for single row
        if labels[2] == 1 or labels[1] == 1 or labels[0] == 1:
            if np.any(labels[3:] > 0):
                labels[0] = 0
    else:
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

    return labels


def reference_loss(df):
    if len(df) >= 10000:
        # Downsample to 10000 observations
        df = df.sample(n=10000, random_state=20)

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


def read_sample_subset_data(
    model_domain,
    downsample_top1 = True,
    upsample_below = 1000,
    sample_size = 4,
    verbose = False,
    toyload = False
):
    df, key = read_data(model_domain = model_domain, toyload = toyload)
    df = resample(df, downsample_top1=downsample_top1, upsample_below=upsample_below, verbose=verbose)
    df = subset_to_smaller(df, sample_size=sample_size)

    return df, key


def train_test_val(df, verbose = False, max_testval = 10**5, testval_fraction = 0.05, test_size = 0.5):
    # Note: More test and validation data exists in sepperate files
    # Test/val size limited to 'max_testval' observations
    if df.shape[0]*testval_fraction > max_testval:
        testval_fraction = max_testval / df.shape[0]

    df_train, df_test = train_test_split(df, test_size=testval_fraction, random_state=20)

    if test_size == 0:
        return df_train, df_test # If test_size is zero, then that means data should only be split in train/test

    df_val, df_test = train_test_split(df_test, test_size=test_size, random_state=20)
    if verbose:
        print(f"Train {df_train.shape[0]} / Val {df_val.shape[0]} / Test {df_test.shape[0]}")

    return df_train, df_val, df_test

# Concat_string
# Makes one string with language and then occupational description
def concat_string(occ1, lang):
    occ1 = str(occ1).strip("'[]'") # pylint: disable=E1310
    # Implement random change to lang 'unknown' here:
    cat_sequence = "<s>" + lang + "</s></s>" + occ1 + "</s>"

    return(cat_sequence)


def concat_string_canine(occ1, lang):
    occ1 = str(occ1).strip("'[]'") # pylint: disable=E1310
    # Implement random change to lang 'unknown' here:
    cat_sequence = lang + "[SEP]" + occ1

    return(cat_sequence)


class OCCDataset(Dataset):
    transform_label: Callable
    # Constructor Function
    def __init__(
        self,
        df_path: str,
        n_obs: int,
        tokenizer,
        attacker,
        max_len,
        n_classes,
        index_file_path,
        formatter: BlockyHISCOFormatter | None = None,
        alt_prob = 0,
        insert_words = False,
        unk_lang_prob = 0.25, # Probability of changing lang to 'unknown'
        model_domain = "",
        new_attack = False # New attack method
    ):
        self.df_path = df_path
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.attacker = attacker
        self.alt_prob = alt_prob # Probability of text alteration in Attacker()
        self.insert_words = insert_words # Should random word insertation occur in Attacker()
        self.unk_lang_prob = unk_lang_prob
        self.model_domain = model_domain
        self.n_obs = n_obs
        self.n_classes = n_classes
        self.formatter = formatter
        self.colnames = pd.read_csv(df_path, nrows=10).columns.tolist()
        self.new_attack = new_attack

        self.setup_target_formatter()

        # Make sure dir exists
        if not os.path.exists(os.path.dirname(df_path)):
            os.makedirs(os.path.dirname(df_path))

        # Load and store index file in memory
        with open(index_file_path, 'r', encoding='utf-8') as index_file:
            self.index_data = index_file.readlines()

    def setup_target_formatter(self):
        if self.formatter is None:
            self.transform_label = partial(labels_to_bin, max_value=self.n_classes)
        else:
            self.transform_label = self.formatter.transform_label

    def __len__(self):
        return self.n_obs

    def __getitem__(self, item):
        # Get the byte offset from the stored index data
        byte_offset_line = self.index_data[item].strip()

        if not byte_offset_line:
            raise ValueError(f"Empty line encountered in index file at item {item}")

        byte_offset = int(byte_offset_line)

        # Use the byte offset to read the specific row from the data file
        with open(self.df_path, 'rb') as f:
            f.seek(byte_offset)
            row = f.readline()
            df = pd.read_csv(io.StringIO(row.decode('utf-8')), names=self.colnames)

        occ1 = str(df.occ1.tolist()[0])
        target = self.transform_label(df)
        lang = str(df.lang.tolist()[0])

        # Implement Attack() here
        occ1 = self.attacker.attack(
            occ1,
            alt_prob = self.alt_prob,
            insert_words = self.insert_words,
            use_textattack=self.new_attack # Uses 'TextAttack' attacker
            )

        # Change lanuage to 'unknown' = "unk" in some cases
        # Warning("CANINEOccupationClassifier: pair of sequences: [CLS] A [SEP] B [SEP]")
        if(random.random() < self.unk_lang_prob):
            lang = "unk"

        occ1 = str(occ1).strip("'[]'") # pylint: disable=E1310

        # Implement random change to lang 'unknown' here:
        if self.model_domain == "Multilingual_CANINE":
            cat_sequence = concat_string_canine(occ1, lang)
        else:
            cat_sequence = concat_string(occ1, lang)

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

        if encoding['input_ids'].shape[1] != self.max_len:
            # breakpoint()
            print(cat_sequence+" had shape: "+str(encoding['input_ids'].shape))
            print("This might cause an error")

        return {
            'occ1': cat_sequence,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long),
        }


class OccDatasetV2(Dataset):
    def __init__(
            self,
            fname_data: str,
            fname_index: str,
            formatter: BlockyHISCOFormatter,
            tokenizer: CanineTokenizer,
            max_input_len: int,
            alt_prob: float = 0.0,
            insert_words: bool = False,
            use_textattack: bool = False,
            unk_lang_prob: float = 0.0,
    ):
        self.fname_data = fname_data
        self.formatter = formatter
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.alt_prob = alt_prob
        self.insert_words = insert_words
        self.use_textattack = use_textattack
        self.unk_lang_prob = unk_lang_prob

        self.colnames: pd.Index = pd.read_csv(fname_data, nrows=1).columns
        self.map_item_byte_index = self._setup_mapping(fname_index)
        # FIXME current impl can mistakenly read an 'occ1' value as an int
        # TODO have `self._get_record` pass in coltypes
        # Example value for 'occ1': 42
        # This is then read as an int; since we read 1 row at a time, we
        # may wrongly infer type in such cases

        if use_textattack:
            # attacker = AttackerClass(...)
            # self.attack = attacker.attack
            raise NotImplementedError('Use of `AttackerClass` not supported yet')
        else:
            self.attack = self._no_attack

    def _setup_mapping(self, fname_index: str) -> dict[int, int]:
        # TODO ask Vedel about structure. Is shuffling implemented?
        # Cannot seem to grab expected elements

        with open(fname_index, 'r', encoding='utf-8') as file:
            byte_offsets = file.readlines()

        return {idx: int(offset) for idx, offset in enumerate(byte_offsets)}

    def _get_record(self, item: int) -> pd.Series:
        with open(self.fname_data, 'rb') as file:
            file.seek(self.map_item_byte_index[item])
            row = file.readline()
            data = pd.read_csv(
                io.StringIO(row.decode('utf-8')),
                names=self.colnames,
                dtype={'occ1': str}, # TODO define full dtype-mapping in self.__init__
                )

        return data.iloc[0]

    def _no_attack(
            self,
            x_string: str,
            alt_prob: float, # pylint: disable=W0613
            insert_words: bool, # pylint: disable=W0613
            use_textattack: bool, # pylint: disable=W0613
    )  -> str:
        ''' Fallback option when `AttackerClass` not available
        '''
        return x_string

    def _prepare_input(self, occ_descr: str, lang: str) -> str:
        occ_descr: str = self.attack( # TODO pass in params at construction since fixed? No need to feed as input each call
            x_string=occ_descr,
            alt_prob=self.alt_prob,
            insert_words=self.insert_words,
            use_textattack=self.use_textattack,
        ) # FIXME need to fix output type. Seems could be str or list[str] as is
        occ_descr = occ_descr.strip("'[]'") # TODO should probably happen before we augment

        if random.random() < self.unk_lang_prob:
            lang = 'unk'

        return '[SEP]'.join((lang, occ_descr)) # TODO '[SEP]' should be a (global) const

    def __len__(self) -> int:
        return len(self.map_item_byte_index)

    def __getitem__(self, item: int) -> dict[str, str | Tensor]:
        record = self._get_record(item)

        occ_descr: str = record.occ1
        lang: str = record.lang
        target = self.formatter.transform_label(record)

        # Augment occupational description and language and
        # return '<LANG>[SEP]<OCCUPATIONAL DESCRIPTION>'
        input_seq = self._prepare_input(occ_descr, lang)

        # Encode input sequence
        encoded_input_seq = self.tokenizer.encode_plus(
            input_seq,
            add_special_tokens=True,
            padding='max_length',
            max_length=self.max_input_len,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',
            truncation = True
        )

        batch_data = {
            'occ1': input_seq, # Legacy name of 'input_seq'
            'input_seq': input_seq,
            'input_ids': encoded_input_seq['input_ids'].flatten(),
            'attention_mask': encoded_input_seq['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long),
        }

        return batch_data


class OccDatasetV2InMem(OccDatasetV2):
    def __init__(
            self,
            fname_data: str,
            fname_index: str,
            formatter: BlockyHISCOFormatter,
            tokenizer: CanineTokenizer,
            max_input_len: int,
            alt_prob: float = 0.0,
            insert_words: bool = False,
            use_textattack: bool = False,
            unk_lang_prob: float = 0.0,
    ):
        super().__init__(
            fname_data=fname_data,
            fname_index=fname_index,
            formatter=formatter,
            tokenizer=tokenizer,
            max_input_len=max_input_len,
            alt_prob=alt_prob,
            insert_words=insert_words,
            use_textattack=use_textattack,
            unk_lang_prob=unk_lang_prob,
        )
        self.frame = pd.read_csv(
            fname_data,
            usecols=['occ1', 'lang', 'code1', 'code2', 'code3', 'code4', 'code5'],
        )

    def _setup_mapping(self, fname_index: str) -> dict[int, int]:
        ''' We avoid using any mapping when loading dataset into memory,
        hence overwrite with ghost method
        '''
        return {}

    def _get_record(self, item: int) -> pd.Series:
        return self.frame.iloc[item]

    def __len__(self) -> int:
        return len(self.frame)


def datasets(
        n_obs_train: int,
        n_obs_val: int,
        n_obs_test: int,
        tokenizer,
        attacker,
        max_len,
        n_classes,
        alt_prob,
        insert_words,
        model_domain,
        formatter: BlockyHISCOFormatter | None = None,
        ): # FIXME move to datasets.py and avoid hardcoded paths
    '''Function to return datasets
    '''
    # File paths for the index files
    train_index_path = "../Data/Tmp_train/Train_index.txt"
    val_index_path = "../Data/Tmp_train/Val_index.txt"
    test_index_path = "../Data/Tmp_train/Test_index.txt"

    # Instantiating OCCDataset with index file paths
    ds_train = OCCDataset(df_path="../Data/Tmp_train/Train.csv", n_obs=n_obs_train, tokenizer=tokenizer, attacker=attacker, max_len=max_len, n_classes=n_classes, index_file_path=train_index_path, alt_prob=0, insert_words=False, model_domain=model_domain, formatter=formatter)
    ds_train_attack = OCCDataset(df_path="../Data/Tmp_train/Train.csv", n_obs=n_obs_train, tokenizer=tokenizer, attacker=attacker, max_len=max_len, n_classes=n_classes, index_file_path=train_index_path, alt_prob=alt_prob, insert_words=insert_words, model_domain=model_domain, formatter=formatter)
    ds_val = OCCDataset(df_path="../Data/Tmp_train/Val.csv", n_obs=n_obs_val, tokenizer=tokenizer, attacker=attacker, max_len=max_len, n_classes=n_classes, index_file_path=val_index_path, alt_prob=0, insert_words=False, model_domain=model_domain, formatter=formatter)
    ds_test = OCCDataset(df_path="../Data/Tmp_train/Test.csv", n_obs=n_obs_test, tokenizer=tokenizer, attacker=attacker, max_len=max_len, n_classes=n_classes, index_file_path=test_index_path, alt_prob=0, insert_words=False, model_domain=model_domain, formatter=formatter)

    return ds_train, ds_train_attack, ds_val, ds_test


def create_data_loader(ds_train, ds_train_attack, ds_val, ds_test, batch_size):
    data_loader_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,  # Enable shuffling
        num_workers=0
    )

    data_loader_train_attack = DataLoader(
        ds_train_attack,
        batch_size=batch_size,
        shuffle=True,  # Enable shuffling
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

# Save tmp_train
def create_index_file(csv_file_path, index_file_path):
    byte_offset = 0
    with open(csv_file_path, 'rb') as f, open(index_file_path, 'w') as index_file:
        next(f)  # Skip the header row
        byte_offset = f.tell()  # Get the current position after skipping
        for line in f:
            index_file.write(f"{byte_offset}\n")
            byte_offset += len(line)


# Saves temporary data to call in training
def save_tmp(df_train, df_val, df_test, path = "../Data/Tmp_train/", verbose = True):
    # Define the directory path
    directory_path = path

    # Check if the directory exists, and create it if not
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    # Define file paths
    train_file_path = os.path.join(directory_path, "Train.csv")
    val_file_path = os.path.join(directory_path, "Val.csv")
    test_file_path = os.path.join(directory_path, "Test.csv")

    # Save the DataFrames to CSV
    df_train.to_csv(train_file_path, index=False)
    df_val.to_csv(val_file_path, index=False)
    df_test.to_csv(test_file_path, index=False)

    # Create index files for each CSV
    create_index_file(train_file_path, os.path.join(directory_path, "Train_index.txt"))
    create_index_file(val_file_path, os.path.join(directory_path, "Val_index.txt"))
    create_index_file(test_file_path, os.path.join(directory_path, "Test_index.txt"))

    if verbose:
        print(f"Saved tmp files to {path}")


# Load data
def load_data(
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
        tokenizer = "No tokenizer", # If no tokenizer is provided one will be created
        model_size = "", # base, large, etc (many transformers have a small and large version)
        formatter: BlockyHISCOFormatter | None = None,
        ):
    # Load data
    df, key = read_sample_subset_data(
        model_domain,
        downsample_top1 = downsample_top1,
        upsample_below = upsample_below,
        sample_size = sample_size,
        verbose = verbose,
        toyload = toyload
        )
    df_train, df_val, df_test = train_test_val(df, verbose=verbose) # pylint: disable=W0632

    # To use later
    n_obs_train = df_train.shape[0]
    n_obs_val = df_val.shape[0]
    n_obs_test = df_test.shape[0]

    # Get set of unique languages
    langs = set(df_train['lang'])
    occs = set(df_train['occ1'])

    # Save tmp files
    save_tmp(df_train, df_val, df_test)

    # Downsample if larger than 100k
    if df.shape[0] > 10**5:
        df = df.sample(n = 10**5, random_state=20)
        df_train = df_train.sample(n=10**5, random_state=20)

    # Load tokenizer (if non is provided)
    if tokenizer == "No tokenizer":
        tokenizer = load_tokenizer(
            model_domain,
            model_size = model_size
            )

        if model_domain != "Multilingual_CANINE":
            tokenizer = update_tokenizer(tokenizer, df_train)

    # Calculate number of classes
    n_classes = len(key)

    # Define attakcer instance
    attacker = AttackerClass(df)

    # Calculate reference loss
    reference_loss_val = reference_loss(df)
    # reference_loss = 0

    # Datsets
    ds_train, ds_train_attack, ds_val, ds_test = datasets(
        n_obs_train, n_obs_val, n_obs_test,
        tokenizer=tokenizer,
        attacker=attacker,
        max_len=max_len,
        n_classes=n_classes,
        alt_prob = alt_prob,
        insert_words = insert_words,
        model_domain = model_domain,
        formatter=formatter,
        )

    # Data loaders
    data_loader_train, data_loader_train_attack, data_loader_val, data_loader_test = create_data_loader(
        ds_train, ds_train_attack, ds_val, ds_test,
        batch_size = batch_size
        )

    return { # TODO this should be a dataclass
        'data_loader_train': data_loader_train,
        'data_loader_train_attack': data_loader_train_attack,
        'data_loader_val': data_loader_val,
        'data_loader_test': data_loader_test,
        'tokenizer': tokenizer,
        'N_CLASSES': n_classes,
        'key': key,
        'reference_loss': reference_loss_val,
        'Languages': langs,
        'Occupations': occs
    }


def load_val(model_domain, sample_size, toyload = False):
    ''' Simple loader for validation data
    '''
    df, key = read_data(model_domain, data_type = "Validation", toyload = toyload)

    # Make id unique by keeping the first occurrence of each id
    n_before = df.shape[0]
    df = df.groupby('RowID').first().reset_index()
    n_after = df.shape[0]

    if n_before != n_after:
        print(f"NOTE: Made data unique for each RowID. Removed {n_before - n_after} observations")

    # Replace 'ge' with 'de' in the 'lang' column
    df['lang'] = df['lang'].replace('ge', 'de')
    print("NOTE: Replaced 'ge' with 'de'. This is a problem stemming from data cleaning.")

    # Subset to smaller
    df = subset_to_smaller(df, sample_size=sample_size)

    # Make binary output matrix
    # df_bin = labels_to_bin(df, max(df.code1)+1)

    return key, df
