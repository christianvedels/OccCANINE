# -*- coding: utf-8 -*-
"""
Train XML Roberta
"""
import os
script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)

#%% Hyperparameters

# Which training data is used for the model
# MODEL_DOMAIN = "HSN_DATABASE"
# MODEL_DOMAIN = "DK_CENSUS"
# MODEL_DOMAIN = "EN_MARR_CERT"
MODEL_DOMAIN = "Multilingual"

# Parameters
SAMPLE_SIZE = 10 # 10 # 10 to the power of this is used for training
EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 2*10**-5
UPSAMPLE_MINIMUM = 0
ALT_PROB = 0.1
INSERT_WORDS = True
DROPOUT_RATE = 0 # Dropout rate in final layer
MAX_LEN = 64 # Number of tokens to use
model_size = "large"


MODEL_NAME = f'XML_RoBERTa_{model_size}_{MODEL_DOMAIN}_sample_size_{SAMPLE_SIZE}_lr_{LEARNING_RATE}_batch_size_{BATCH_SIZE}' 
checkpoint_path = None
# checkpoint_path = "../Trained_models/231111_XML_RoBERTa_Multilingual_sample_size_6_lr_2e-05_batch_size_32" # Provide path to load model from checkpoint path

#%% Libraries
# Import necessary libraries
import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report

#%% Load modules
from n001_Model_assets import XMLRoBERTaOccupationClassifier
from n101_Trainer import trainer_loop
from n102_DataLoader import Load_data, load_model_from_checkpoint

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
    verbose = False,
    model_size = model_size
    # , toyload=True
    )

# Sanity check
# for d in data['data_loader_train_attack']: 
#     print(d['occ1'][0])
    
# data['reference_loss']

# %% Load model
model = XMLRoBERTaOccupationClassifier(
    model_domain = MODEL_DOMAIN,
    n_classes = data['N_CLASSES'], 
    tokenizer = data['tokenizer'], 
    dropout_rate = DROPOUT_RATE,
    model_size = model_size
    )
model.to(device)

# %% Load model checkpoint
if checkpoint_path:
    model, tokenizer = load_model_from_checkpoint(checkpoint_path, model, MODEL_DOMAIN)
    data['tokenizer'] = tokenizer

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







