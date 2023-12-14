# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 10:20:17 2023
https://www.kaggle.com/code/mehmetlaudatekman/lstm-text-classification-pytorch
@author: chris
"""

# -*- coding: utf-8 -*-
"""
Train Character Architecture with No tokenization In Neural Encoders
CANINE
"""
import os
script_directory = os.path.dirname(os.path.abspath(__name__))
os.chdir(script_directory)

#%% Hyperparameters

# Choose which GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Which training data is used for the model
MODEL_DOMAIN = "Multilingual_CANINE"

# Parameters
SAMPLE_SIZE = 3 # 10 # 10 to the power of this is used for training
EPOCHS = 500
BATCH_SIZE = 256
LEARNING_RATE = 2*10**-5
UPSAMPLE_MINIMUM = 0
ALT_PROB = 0.1
INSERT_WORDS = True
DROPOUT_RATE = 0 # Dropout rate in final layer
MAX_LEN = 128 # Number of tokens/characters to use

MODEL_NAME = f'CANINE_{MODEL_DOMAIN}_sample_size_{SAMPLE_SIZE}_lr_{LEARNING_RATE}_batch_size_{BATCH_SIZE}' 

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
# device = "cpu"

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
    # , toyload=True
    )

# # Sanity check
# for d in data['data_loader_train_attack']: 
#     print(d['occ1'][0])
#     print(data['tokenizer'](d['occ1'][0]))
    
# data['reference_loss']

# %% Load model
model = CANINEOccupationClassifier(
    model_domain = MODEL_DOMAIN,
    n_classes = data['N_CLASSES'], 
    tokenizer = data['tokenizer'], 
    dropout_rate = DROPOUT_RATE
    )
model.to(device)

# %% Sanity check
# in0 = data["tokenizer"](
#     ["masion", "mason"], 
#     padding = 'max_length',
#     max_length = MAX_LEN,
#     return_token_type_ids=False,
#     return_attention_mask=True,
#     return_tensors='pt',
#     truncation = True
#     )

# model.forward(in0['input_ids'], in0['attention_mask'])
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







