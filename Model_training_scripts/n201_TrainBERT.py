# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 13:34:23 2023
This script fine tunes a BERT model to classify occupational descriptions as 



@author: chris
"""
import os
script_directory = os.path.dirname(os.path.abspath(__name__))
os.chdir(script_directory)

#%% Hyperparameters

# Which training data is used for the model
# MODEL_DOMAIN = "HSN_DATABASE"
MODEL_DOMAIN = "DK_CENSUS"
#MODEL_DOMAIN = "EN_MARR_CERT"

# Parameters
SAMPLE_SIZE = 5 # 10 to the power of this is used for training
EPOCHS = 500
BATCH_SIZE = 2**5
LEARNING_RATE = 2*10**-5
UPSAMPLE_MINIMUM = 0
ALT_PROB = 0.1
INSERT_WORDS = True
DROPOUT_RATE = 0 # Dropout rate in final layer
MAX_LEN = 50 # Number of tokens to use

MODEL_NAME = f'BERT_{MODEL_DOMAIN}_sample_size_{SAMPLE_SIZE}_lr_{LEARNING_RATE}_batch_size_{BATCH_SIZE}' 

#%% Libraries
# Import necessary libraries
import numpy as np
import pandas as pd
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report

#%% Load modules
from n001_Model_assets import *
from n100_Attacker import *
from n101_Trainer import *
from n102_DataLoader import *

#%% Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% Load data + tokenizer
data = Load_data(
    model_domain = MODEL_DOMAIN,
    downsample_top1 = True,
    upsample_below = UPSAMPLE_MINIMUM,
    sample_size = SAMPLE_SIZE,
    max_len = MAX_LEN,
    alt_prob = ALT_PROB,
    insert_words = INSERT_WORDS,
    batch_size = BATCH_SIZE,
    verbose = False
    )

# # Sanity check
# for d in data['data_loader_train_attack']: 
#    print(d['occ1'][0][0])
    
# data['reference_loss']

# %% Load model
model = BERTOccupationClassifier(
    n_classes = data['N_CLASSES'], 
    model_domain = MODEL_DOMAIN, 
    tokenizer = data['tokenizer'], 
    dropout_rate = DROPOUT_RATE
    )
model.to(device)

#%% Optimizer and learning scheduler
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = len(data['data_loader_train']) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)
# Set the loss function 
loss_fn = nn.BCEWithLogitsLoss().to(device)

# %%
model = trainer_loop(
    model = model, 
    epochs = EPOCHS, 
    model_name = MODEL_NAME, 
    data = data, 
    loss_fn = loss_fn, 
    reference_loss = data['reference_loss'], 
    optimizer = optimizer, 
    device = device, 
    scheduler = scheduler
    )


# %% Load best model instance
# Define the path to the saved binary file
model_path = '../Trained_models/'+MODEL_NAME+'.bin'

# Load the model
loaded_state = torch.load(model_path)
model_best = BERTOccupationClassifier(
    n_classes = data['N_CLASSES'], 
    model_domain = MODEL_DOMAIN, 
    tokenizer = data['tokenizer'], 
    dropout_rate = DROPOUT_RATE
    )
model_best.load_state_dict(loaded_state)
model_best.to(device)

# %%
y_occ_texts, y_pred, y_pred_probs, y_test = get_predictions(
    model_best,
    data['data_loader_test'],
    device = device
)
report = classification_report(y_test, y_pred, output_dict=True)

print_report(report, MODEL_NAME)





