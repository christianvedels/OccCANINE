# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 21:20:40 2023

This script retrains the model with e.g. additional data

@author: chris
"""
# -*- coding: utf-8 -*-
"""
Train XML Roberta
"""
import os
script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)

#%% Hyperparameters

# Which training data is used for the model
MODEL_DOMAIN = "Multilingual_CANINE"

# Parameters
SAMPLE_SIZE = 6 # 10 to the power of this is used for training
EPOCHS = 30
BATCH_SIZE = 2**5
LEARNING_RATE = 2*10**-5
UPSAMPLE_MINIMUM = 0
ALT_PROB = 0.1
INSERT_WORDS = True
DROPOUT_RATE = 0 # Dropout rate in final layer
MAX_LEN = 64 # Number of tokens to use

import datetime
current_date = datetime.datetime.now().strftime("%y%m%d")
if MODEL_DOMAIN == "Multilingual":
    MODEL_NAME_start = f'231107_XML_RoBERTa_{MODEL_DOMAIN}_sample_size_{SAMPLE_SIZE}_lr_{LEARNING_RATE}_batch_size_{BATCH_SIZE}' 
    MODEL_NAME = f'{current_date}_XML_RoBERTa_{MODEL_DOMAIN}_sample_size_{SAMPLE_SIZE}_lr_{LEARNING_RATE}_batch_size_{BATCH_SIZE}'
elif MODEL_DOMAIN == "Multilingual_CANINE":
    MODEL_NAME_start = "CANINE_Multilingual_CANINE_sample_size_6_lr_2e-05_batch_size_32"
    MODEL_NAME_start = "231114_231114_231114_CANINE_Multilingual_CANINE_sample_size_6_lr_2e-05_batch_size_32"
    MODEL_NAME = f"{current_date}_CANINE_Multilingual_CANINE_sample_size_6_lr_2e-05_batch_size_32"
else: 
    raise Exception("Not implemented")
    MODEL_NAME = f'{current_date}_BERT_{MODEL_DOMAIN}_sample_size_{SAMPLE_SIZE}_lr_{LEARNING_RATE}_batch_size_{BATCH_SIZE}' 

#%% Libraries
# Import necessary libraries
import numpy as np
import pandas as pd
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report
from transformers import AutoTokenizer

#%% Load modules
from n001_Model_assets import *
from n100_Attacker import *
from n101_Trainer import *
from n102_DataLoader import *

#%% Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% Load tokenizer

if MODEL_DOMAIN == "Multilingual":
   # Load updated tokenizer
   tokenizer_save_path = '../Trained_models/' + MODEL_NAME_start + '_tokenizer'
   tokenizer = AutoTokenizer.from_pretrained(tokenizer_save_path)
elif MODEL_DOMAIN == "Multilingual_CANINE":
    tokenizer = load_tokenizer(MODEL_DOMAIN)
else: 
    raise Exception("Not implemented")


# Temp code 
# tokenizer = data['tokenizer']
test = "This is a sentence"

tokenizer(test)

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
    verbose = False,
    # toyload=True,
    tokenizer=tokenizer
    )

# %% Load best model instance
# Define the path to the saved binary file
model_path = '../Trained_models/'+MODEL_NAME_start+'.bin'

# Load the model
loaded_state = torch.load(model_path)

model = XMLRoBERTaOccupationClassifier(
    n_classes = len(data['key']), 
    model_domain = MODEL_DOMAIN, 
    tokenizer = tokenizer, 
    dropout_rate = DROPOUT_RATE
    )
    
model.load_state_dict(loaded_state)

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
    scheduler = scheduler,
    attack_switch = True  
    )

# # %% Load best model instance
# # Define the path to the saved binary file
# model_path = '../Trained_models/'+MODEL_NAME+'.bin'

# # Load the model
# loaded_state = torch.load(model_path)
# model_best = XMLRoBERTaOccupationClassifier(
#     n_classes = data['N_CLASSES'], 
#     model_domain = MODEL_DOMAIN, 
#     tokenizer = data['tokenizer'], 
#     dropout_rate = DROPOUT_RATE
#     )
# model_best.load_state_dict(loaded_state)
# model_best.to(device)

# # %%
# y_occ_texts, y_pred, y_pred_probs, y_test = get_predictions(
#     model_best,
#     data['data_loader_test'],
#     device = device
# )
# report = classification_report(y_test, y_pred, output_dict=True)

# print_report(report, MODEL_NAME)






