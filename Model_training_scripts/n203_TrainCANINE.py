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

import wandb
os.environ['WANDB_API_KEY'] = '30b543a8072aecce8ff8922a0f68e0785f237d92'  # Replace with your actual API key
wandb.login()
wandb.init(project='Multilingual_CANINE', entity='christianmdahl')

# Which training data is used for the model
MODEL_DOMAIN = "Multilingual_CANINE"

# Parameters
SAMPLE_SIZE = 10 # 10 to the power of this is used for training
EPOCHS = 500
BATCH_SIZE = 256
LEARNING_RATE = 2*10**-5
UPSAMPLE_MINIMUM = 0
ALT_PROB = 0.1
INSERT_WORDS = True
DROPOUT_RATE = 0 # Dropout rate in final layer
MAX_LEN = 128 # Number of tokens/characters to use

MODEL_NAME = f'CANINE_{MODEL_DOMAIN}_sample_size_{SAMPLE_SIZE}_lr_{LEARNING_RATE}_batch_size_{BATCH_SIZE}' 

# checkpoint_path = None # Provide path to load model from checkpoint path
checkpoint_path = "../Trained_models/CANINE_Multilingual_CANINE_sample_size_10_lr_2e-05_batch_size_256"

# Configure hyperparameters
config = wandb.config
config.learning_rate = LEARNING_RATE
config.epochs = EPOCHS
config.batch_size = BATCH_SIZE
config.dropout_rate = DROPOUT_RATE
config.max_len = MAX_LEN
config.sample_size = SAMPLE_SIZE
config.alt_prob = ALT_PROB
config.insert_words = INSERT_WORDS
config.model_domain = MODEL_DOMAIN
config.upsample_minimum = UPSAMPLE_MINIMUM
config.model_name = MODEL_NAME
config.checkpoint_path = checkpoint_path

#%% Libraries
# Import necessary libraries
import torch
from torch import nn
from transformers import AdamW, get_linear_schedule_with_warmup

#%% Load modules
from n001_Model_assets import CANINEOccupationClassifier, load_model_from_checkpoint
from n101_Trainer import trainer_loop
from n102_DataLoader import Load_data

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

# %% Load model
model = CANINEOccupationClassifier(
    model_domain = MODEL_DOMAIN,
    n_classes = data['N_CLASSES'], 
    tokenizer = data['tokenizer'], 
    dropout_rate = DROPOUT_RATE
    )
model.to(device)

# %% Load model checkpoint
if checkpoint_path:
    model, tokenizer = load_model_from_checkpoint(checkpoint_path, model, MODEL_DOMAIN)
    data['tokenizer'] = tokenizer

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
wandb.finish()








# %%
