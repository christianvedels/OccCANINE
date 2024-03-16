# -*- coding: utf-8 -*-
"""
Train Character Architecture with No tokenization In Neural Encoders
CANINE
"""


import argparse
import os

import torch

from torch import nn
from transformers import AdamW, get_linear_schedule_with_warmup

# Load modules
from hisco.model_assets import CANINEOccupationClassifier, load_model_from_checkpoint
from hisco.trainer import trainer_loop
from hisco.dataloader import load_data

# Hyperparameters

# Choose which GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Which training data is used for the model
MODEL_DOMAIN = "Multilingual_CANINE"

# Parameters
SAMPLE_SIZE = 10 # 10 to the power of this is used for training
EPOCHS = 50
BATCH_SIZE = 256
LEARNING_RATE = 2*10**-5
UPSAMPLE_MINIMUM = 0
ALT_PROB = 0.1
INSERT_WORDS = True
DROPOUT_RATE = 0 # Dropout rate in final layer
MAX_LEN = 128 # Number of tokens/characters to use

MODEL_NAME = f'CANINE_{MODEL_DOMAIN}_sample_size_{SAMPLE_SIZE}_lr_{LEARNING_RATE}_batch_size_{BATCH_SIZE}'

# checkpoint_path = None # Provide path to load model from checkpoint path
CHECKPOINT_PATH = "../Trained_models/CANINE_Multilingual_CANINE_sample_size_4_lr_2e-05_batch_size_128" # FIXME bad hardcode


def parse_args() -> argparse.Namespace:
    raise NotImplementedError


def main():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #  Load data + tokenizer
    data = load_data(
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

    #  Load model
    model = CANINEOccupationClassifier(
        model_domain=MODEL_DOMAIN,
        n_classes=data['N_CLASSES'],
        # tokenizer = data['tokenizer'], # FIXME this is not a valid arg?
        dropout_rate=DROPOUT_RATE,
        )
    model.to(device)

    #  Load model checkpoint
    if CHECKPOINT_PATH:
        model, tokenizer = load_model_from_checkpoint(CHECKPOINT_PATH, model, MODEL_DOMAIN)
        data['tokenizer'] = tokenizer

    # Optimizer and learning scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(data['data_loader_train']) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # Set the loss function
    loss_fn = nn.BCEWithLogitsLoss().to(device)

    model = trainer_loop( # FIXME why is there an assignment here?
        model=model,
        epochs=EPOCHS,
        model_name=MODEL_NAME,
        data=data,
        loss_fn=loss_fn,
        reference_loss=data['reference_loss'],
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        )


if __name__ == '__main__':
    main()
