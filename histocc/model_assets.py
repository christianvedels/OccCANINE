# -*- coding: utf-8 -*-
"""
Models
Created on Tue May 23 15:02:16 2023

Authors: Christian Vedel, [christian-vs@sam.sdu.dk],

Purpose: Defines model classes
"""


import torch

import torch.nn as nn

from huggingface_hub import PyTorchModelHubMixin

from transformers import (
    XLMRobertaTokenizer,
    CanineTokenizer,
    CanineModel,
    AutoTokenizer,
)

# Model path from domain
def modelPath(model_domain, model_size = ""):
    if model_domain == "DK_CENSUS":
        mdl = 'Maltehb/danish-bert-botxo' # https://huggingface.co/Maltehb/danish-bert-botxo
    elif model_domain == "EN_MARR_CERT":
        mdl = "bert-base-uncased"
    elif model_domain == "HSN_DATABASE":
        mdl = "GroNLP/bert-base-dutch-cased"
    elif model_domain == "Multilingual":
        if model_size == "base":
            mdl = 'xlm-roberta-base'
        elif model_size == "large":
            mdl = 'xlm-roberta-large'
    elif model_domain == "Multilingual_CANINE":  # Replace "CANINE_MODEL_NAME" with the actual CANINE model name
        mdl = "google/canine-s"
    else:
        raise NotImplementedError("This is not implemented yet")

    return mdl


# Tokenizer
def load_tokenizer(model_domain, model_size = ""):
    mdl = modelPath(model_domain, model_size)

    if mdl in ("xlm-roberta-base", "xlm-roberta-large"):
        tokenizer = XLMRobertaTokenizer.from_pretrained(mdl)
    elif mdl == "google/canine-s":  # Replace "CANINE_MODEL_NAME" with the actual CANINE model name
        tokenizer = CanineTokenizer.from_pretrained(mdl)
    else:
        # Consider implementing this
        # tokenizer = BertTokenizer.from_pretrained(MDL)
        raise NotImplementedError("Not implemented")

    return tokenizer


# Update tokenizer
def update_tokenizer(tokenizer, df):
    # Add unseen words
    all_text = ' '.join(df.occ1.tolist())
    words_list = all_text.split()
    unique_words = set(words_list)
    all_lang_words = set(df.lang)
    unique_words.update(all_lang_words)
    unique_words.update("unk") # Unknown language token
    # Add tokens for missing words
    tokenizer.add_tokens(list(unique_words))

    return tokenizer

#
def getModel(model_domain, model_size = ""):
    # breakpoint()
    mdl = modelPath(model_domain, model_size)

    if model_domain == "Multilingual_CANINE":
        model = CanineModel.from_pretrained(mdl)
    else:
        # model = BertModel.from_pretrained(MDL)
        raise NotImplementedError("Not implemented")

    return model


# Build the Classifier
class CANINEOccupationClassifier(nn.Module):
    # Constructor class
    def __init__(self, n_classes, model_domain, dropout_rate):
        super(CANINEOccupationClassifier, self).__init__()
        self.basemodel = getModel(model_domain)
        self.drop = nn.Dropout(p=dropout_rate)
        self.out = nn.Linear(self.basemodel.config.hidden_size, n_classes)

    def resize_token_embeddings(self, n):
        pass # Do nothing CANINE should never be resized

    # Forward propagaion class
    def forward(self, input_ids, attention_mask):
        outputs = self.basemodel(
          input_ids=input_ids,
          attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output

        #  Add a dropout layer
        output = self.drop(pooled_output)
        return self.out(output)


# Build the Classifier for HF hub
class CANINEOccupationClassifier_hub(nn.Module, PyTorchModelHubMixin):
    # Constructor class
    def __init__(self, config):
        super().__init__()
        self.basemodel = getModel(config["model_domain"])
        self.drop = nn.Dropout(p=config["dropout_rate"])
        self.out = nn.Linear(self.basemodel.config.hidden_size, config["n_classes"])

    def resize_token_embeddings(self, n):
        pass # Do nothing CANINE should never be resized

    # Forward propagaion class
    def forward(self, input_ids, attention_mask):
        outputs = self.basemodel(
          input_ids=input_ids,
          attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output

        #  Add a dropout layer
        output = self.drop(pooled_output)
        return self.out(output)


# Load model from checkpoint
def load_model_from_checkpoint(checkpoint_path, model, model_domain):
    # Handle string
    if checkpoint_path.endswith(".bin"):
        checkpoint_path = checkpoint_path[:-4]  # Remove the ".bin" extension

    if model_domain == "Multilingual":
        # Load updated tokenizer
        tokenizer_save_path = checkpoint_path + '_tokenizer'
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_save_path)
    elif model_domain == "Multilingual_CANINE":
        tokenizer = load_tokenizer(model_domain)
    else:
        raise NotImplementedError("Not implemented")

    # Adapt model size to the tokenizer size:
    model.resize_token_embeddings(len(tokenizer))

    # Load model state
    loaded_state = torch.load(checkpoint_path+".bin")
    model.load_state_dict(loaded_state)

    return model, tokenizer
