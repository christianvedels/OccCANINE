# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 22:57:01 2023

@author: chris
"""
import os
script_directory = os.path.dirname(os.path.abspath(__name__))
os.chdir(script_directory)

#%% Hyperparameters

# Which training data is used for the model
MODEL_DOMAIN = "Ensemble"

# Parameters
SAMPLE_SIZE = 6 # 10 to the power of this is used for training
EPOCHS = 30
BATCH_SIZE = 2**5
LEARNING_RATE = 2*10**-5
UPSAMPLE_MINIMUM = 0
ALT_PROB = 0
INSERT_WORDS = True
DROPOUT_RATE = 0 # Dropout rate in final layer

import datetime
current_date = datetime.datetime.now().strftime("%y%m%d")
start_models = [
    '231111_XML_RoBERTa_Multilingual_sample_size_6_lr_2e-05_batch_size_32',
    '231115_CANINE_Multilingual_CANINE_sample_size_6_lr_2e-05_batch_size_32'
    
    ]

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
from n103_Model_loader import *

#%% Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"

# %% Models
models = [Finetuned_model(x, device) for x in start_models]

# %% Sanity check
x = ["tailor of fine dresses"]
models[0].predict(x, what = "pred", lang = "en")
models[1].predict(x, what = "pred", lang = "en")
# models[0].predict(x, what = 3, lang = "en")
# models[1].predict(x, what = 3, lang = "en")

# %% Load data + tokenizer
key, df, df_bin = Load_val(
    model_domain = "Multilingual",
    sample_size = 4 # SAMPLE_SIZE
    )

# %% Run test
pred00 = models[0].predict(df["occ1"].tolist(), what = "bin", lang = df['lang'].tolist(), verbose = True)
pred01 = models[0].predict(df["occ1"].tolist(), what = "bin", verbose = True)
pred10 = models[1].predict(df["occ1"].tolist(), what = "bin", lang = df['lang'].tolist(), verbose = True)
pred11 = models[1].predict(df["occ1"].tolist(), what = "bin", verbose = True)




# %%
def pred_test(pred):
    test = []
    for p, y in zip(pred, df_bin):
        result = p == y
        test.append(all(result))
        
    return np.mean(test)
    
pred_test(pred00)
pred_test(pred01)
pred_test(pred10)
pred_test(pred10)

# %%
pred025 =  [
    models[0].predict(df["occ1"].tolist(), what = "bin", lang = df['lang'].tolist(), verbose = True, threshold = p)
    for p in np.arange(0.1, 1, step=0.1)
    ]

pred125 =  [
    models[1].predict(df["occ1"].tolist(), what = "bin", lang = df['lang'].tolist(), verbose = True, threshold = p)
    for p in np.arange(0.1, 1, step=0.1)
    ]

# %%
[[pred_test(pred),p] for pred, p in zip(pred025, np.arange(0.1, 1, step=0.1))]

# %% Run test
pred00 = models[0].predict(df["occ1"].tolist(), what = "probs", lang = df['lang'].tolist(), verbose = True)
pred01 = models[0].predict(df["occ1"].tolist(), what = "probs", verbose = True)
pred10 = models[1].predict(df["occ1"].tolist(), what = "probs", lang = df['lang'].tolist(), verbose = True)
pred11 = models[1].predict(df["occ1"].tolist(), what = "probs", verbose = True)

pred00[0].shape
for i, x in enumerate(pred00):
    if(x.shape[0]==64):
        nope = 100
    else:
        print(f"{i}: found it!")

pred00 = np.vstack(pred00)
pred01 = np.vstack(pred01)
pred10 = np.vstack(pred10)
pred11 = np.vstack(pred11)

# Stack the arrays horizontally to create a single 4xKxN array
stacked_array = np.stack((pred00, pred01, pred10, pred11), axis=0)

# Reshape the array to 4x(N*5)
reshaped_array = stacked_array.reshape(4, -1)
# Calculate the correlation matrix
K = pred00.shape[1]
for j in range(K):
    x00 = pred00[:,j]
    x01 = pred00[:,j]
    x10 = pred00[:,j]
    x00 = pred00[:,j]

correlation_matrix = np.corrcoef(reshaped_array)

# %% Define ensemble model
base_models = [m.model.basemodel for m in models]
encoders = [m.encoder for m in models]

# Define a more complex meta-classifier with a non-linear activation
meta_classifier = nn.Sequential(
    nn.Linear(len(base_models) * len(key), 2048),
    nn.ReLU(),
    nn.Linear(2048, len(key))
)

StackingEnsemble(base_models, meta_classifier, freeze_base_models=True)

# Move the ensemble model to the appropriate device
ensemble_model.to(device)

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

# Convert the input data to torch tensors and move them to the device
input_data = torch.tensor(reshaped_array, dtype=torch.float32).to(device)
labels = torch.tensor(df_bin, dtype=torch.long).to(device)

# Training loop
for epoch in range(EPOCHS):
    # Forward pass
    outputs = ensemble_model(input_data)
    loss = criterion(outputs, labels)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Update the learning rate
    scheduler.step()

    # Print the loss every few epochs
    if epoch % 5 == 0:
        print(f'Epoch {epoch}/{EPOCHS}, Loss: {loss.item()}')



