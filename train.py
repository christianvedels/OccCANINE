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
from histocc import (
    CANINEOccupationClassifier,
    load_model_from_checkpoint,
    trainer_loop,
    load_data,
    )


# Choose which GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # TODO promote to arg?

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # File paths, data & model choices
    parser.add_argument('--checkpoint-path', type=str, default=None) # FIXME use "../Trained_models/CANINE_Multilingual_CANINE_sample_size_4_lr_2e-05_batch_size_128"?
    parser.add_argument('--model-name', type=str, default=None) # TODO investigate real purpose of this and why not covered in --checkpoint-path
    parser.add_argument('--model-domain', type=str, default='Multilingual_CANINE') # TODO Why is this called domain when description is "Which training data is used for the model"?

    # Parameters
    parser.add_argument('--sample-size', type=int, default=10) # FIXME this appears to not be actual size but rather a modifier: "10 to the power of this is used for training"
    parser.add_argument('--spochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=2*10**-5)
    parser.add_argument('--upsample-minimum', type=int, default=0)
    parser.add_argument('--alt-prob', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate in final layer')
    parser.add_argument('--max-len', type=int, default=128, help='Number of tokens/characters to use')
    parser.add_argument('--skip-insert-words', action='store_true', default=False)

    args = parser.parse_args()

    if args.model_name is None:
        args.model_name = f'CANINE_{args.model_domain}_sample_size_{args.sample_size}_lr_{args.learning_rate}_batch_size_{args.batch_size}'

    args.insert_words = not args.skip_insert_words

    return args


def main():
    args = parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #  Load data + tokenizer
    data = load_data(
        model_domain=args.model_domain,
        downsample_top1=True,
        upsample_below=args.upsample_minimum,
        sample_size=args.sample_size,
        max_len=args.max_len,
        alt_prob=args.alt_prob,
        insert_words=args.insert_words,
        batch_size=args.batch_size,
        verbose=False,
        )

    #  Load model
    model = CANINEOccupationClassifier(
        model_domain=args.model_domain,
        n_classes=data['N_CLASSES'],
        dropout_rate=args.dropout_rate,
        )
    model.to(device)

    #  Load model checkpoint
    if args.checkpoint_path is not None:
        model, tokenizer = load_model_from_checkpoint(args.checkpoint_path, model, args.model_domain)
        data['tokenizer'] = tokenizer

    # Optimizer and learning scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(data['data_loader_train']) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # Set the loss function
    loss_fn = nn.BCEWithLogitsLoss().to(device)

    model = trainer_loop( # FIXME why is there an assignment here?
        model=model,
        epochs=args.epochs,
        model_name=args.model_name,
        data=data,
        loss_fn=loss_fn,
        reference_loss=data['reference_loss'],
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        )


if __name__ == '__main__':
    main()
