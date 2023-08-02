# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 09:58:40 2023

@author: christian-vs
"""

# %% Parameters
n_epochs = 10 # The networks trains a maximum of these epochs

sample_size = 4
batch_size = 16

# %% Import packages
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import os
import random as r
from transformers import AutoTokenizer

# %% Check cuda availability
is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

# device = torch.device("cpu")    

#%% Load data
if os.environ['COMPUTERNAME'] == 'SAM-126260':
    os.chdir('D:/Dropbox/PhD/HISCO clean')
else:
    os.chdir('C:/Users/chris/Dropbox/PhD/HISCO clean')
print(os.getcwd())

fname = "Data/Training_data/DK_census_train.csv"
df = pd.read_csv(fname, encoding = "UTF-8")

# Subset to larger toy data
r.seed(20)
df = df.sample(10**sample_size)


# # %% Resample
# # Count the number of occurrences for each class in 'code1', 'code2', 'code3', 'code4', 'code5'
# all_class_columns = ['code1', 'code2', 'code3', 'code4', 'code5']
# class_counts = df[all_class_columns].stack().value_counts()

# # Identify the classes with occurrences less than 1000 and more than 10% of the data
# classes_less_than_1000 = class_counts.index[class_counts < 1000]
# classes_more_than_10_percent = class_counts.index[class_counts > df.shape[0] * 0.1]

# for i in classes_less_than_1000:
#     i





# # Upsample classes with occurrences less than 1000 to 1000 samples
# for col in ['code1', 'code2', 'code3', 'code4', 'code5']:
#     mask = df[col].isin(classes_less_than_1000)  # Filter rows where the class code is in small classes
#     small_class_indices = df[mask].index.tolist()  # Get indices of rows with small classes
#     upsampled_rows = df.loc[r.choices(small_class_indices, k=1000 - len(small_class_indices))]
#     df = pd.concat([df, upsampled_rows])

# # Downsample classes with occurrences more than 10% to 10% of the data
# for col in ['code1', 'code2', 'code3', 'code4', 'code5']:
#     mask = df[col].isin(classes_more_than_10_percent)  # Filter rows where the class code is in large classes
#     large_class_indices = df[mask].index.tolist()  # Get indices of rows with large classes
#     downsampled_rows = df.loc[r.sample(large_class_indices, k=int(df.shape[0] * 0.1))]
#     df = pd.concat([df, downsampled_rows])

# # Shuffle the dataframe to ensure the order is random
# df = df.sample(frac=1).reset_index(drop=True)


#%% Tokenizer
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')

# Now we need to update the tokenizer with words it has not seen before
# List of unique words
all_text = ' '.join(df.occ1.tolist())
words_list = all_text.split()
unique_words = set(words_list)

# Add tokens for missing words
tokenizer.add_tokens(list(unique_words))

# Testing tokenizer
print({x : tokenizer.encode(x, add_special_tokens=False) for x in df.occ1.values[0].split()})

# Tokenize features
MAX_LEN = 128
features = df.occ1.values.tolist()
tokenized_feature = tokenizer.batch_encode_plus(
                            # Sentences to encode
                            features, 
                            # Add '[CLS]' and '[SEP]'
                            add_special_tokens = True,
                            # Add empty tokens if len(text)<MAX_LEN
                            padding = 'max_length',
                            # Truncate all sentences to max length
                            truncation=True,
                            # Set the maximum length
                            max_length = MAX_LEN, 
                            # Return attention mask
                            return_attention_mask = True,
                            # Return pytorch tensors
                            return_tensors = 'pt'       
                   )

# %% Preprocessesing
tokenizer("huusmand og fisker mand")

# MAX_LEN = 512
MAX_LEN = 64
tokenized_feature = tokenizer.batch_encode_plus(
    # Sentences to encode
    df['occ1'].tolist(),
    # Add '[CLS]' and '[SEP]'
    add_special_tokens = True,
    # Add empty tokens if len(text)<MAX_LEN
    padding = 'max_length',
    # Truncate all sentences to max length
    truncation=True,
    # Set the maximum length
    max_length = MAX_LEN, 
    # Return attention mask
    return_attention_mask = True,
    # Return pytorch tensors
    return_tensors = 'pt'       
    )

# %% Labels to list
# Each person can have up to 5 occupations
df_codes = df[["code1", "code2", "code3", "code4", "code5"]]

# Load binarizer
# from sklearn.preprocessing import MultiLabelBinarizer
# one_hot_labels = MultiLabelBinarizer()

# Binarize
labels_list = df_codes.values.tolist()

# Build outcome matrix
# Convert the given list to a NumPy array
labels_array = np.array(labels_list)

# Find the maximum integer value in the given array
max_value = int(np.nanmax(labels_array))

# Construct the NxK matrix
N = len(labels_list)
K = max_value + 1
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

# %% Prepare data
from sklearn.model_selection import train_test_split
train_inputs, validation_inputs, train_labels, validation_labels, train_masks, validation_masks = train_test_split(
    tokenized_feature['input_ids'], labels, tokenized_feature['attention_mask'], random_state=20, test_size=0.2
    )

# Create the DataLoader for our training set
train_data = TensorDataset(train_inputs, train_masks, torch.tensor(train_labels))
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
# Create the DataLoader for our test set
validation_data = TensorDataset(validation_inputs, validation_masks, torch.tensor(validation_labels))
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

# %% Define model
from transformers import XLMRobertaForSequenceClassification, AdamW, get_linear_schedule_with_warmup
model = XLMRobertaForSequenceClassification.from_pretrained(
    "xlm-roberta-base", 
    # Specify number of classes
    num_labels = labels.shape[1], 
    # Whether the model returns attentions weights
    output_attentions = False,
    # Whether the model returns all hidden-states 
    output_hidden_states = False,
    problem_type="multi_label_classification"
)

# Adapt model size to the tokens added:
model.resize_token_embeddings(len(tokenizer))

# Optimizer & Learning Rate Scheduler
optimizer = AdamW(model.parameters(),
                  lr = 3e-1, 
                  eps = 1e-16
                )

# Number of training epochs
epochs = n_epochs
# Total number of training steps is number of batches * number of epochs.
total_steps = len(df['occ1'].tolist()) * epochs
# Create the learning rate scheduler
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)

# tell pytorch to run this model on GPU
model.cuda()
# model.to('cpu')

# %% Training

# Training
import time
# Store the average loss after each epoch 
loss_values = []

# number of total steps for each epoch
print('total steps per epoch: ',  len(df['occ1'].tolist()) / batch_size)
# looping over epoch
for epoch_i in range(0, epochs):
    
    print('training on epoch: ', epoch_i)
    # set start time 
    t0 = time.time()
    # reset total loss
    total_loss = 0
    # model in training 
    model.train()
    # loop through batch 
    for step, batch in enumerate(train_dataloader):
        # load data from dataloader 
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        # clear any previously calculated gradients 
        model.zero_grad()
        # get outputs
        outputs = model(b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask,
                        labels=b_labels)
        # get loss
        loss = outputs[0]
        # total loss
        total_loss += loss.item()
        # clip the norm of the gradients to 1.0.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # update optimizer
        optimizer.step()
        # update learning rate 
        scheduler.step()
        
        # Progress update every 10 steps
        if step % 10 == 0 and not step == 0:
            print(
                'training on step: ', step, 
                '; loss:', np.round(loss.item(), 4) ,
                # 'total time used is: {0:.2f} s'.format(time.time() - t0),
                end = '\n'
                )
    # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(train_dataloader)
    # Store the loss value for plotting the learning curve..
    loss_values.append(avg_train_loss)
    print("average training loss: {0:.2f}".format(avg_train_loss))