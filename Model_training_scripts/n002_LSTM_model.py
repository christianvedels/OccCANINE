# -*- coding: utf-8 -*-
"""
Models
Created on Tue May 23 15:02:16 2023

Authors: Christian Vedel, [christian-vs@sam.sdu.dk],

Purpose: Defines model classes
https://discuss.huggingface.co/t/character-level-tokenizer/12450/2
"""

# Set path to file path
import os
script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)

# %% Libraries
import numpy as np
import pandas as pd
from transformers import BertModel, BertTokenizer
from torch import nn

#%% Tokenizer
def lstm_tokenizer(model_domain, df):
    # Add unseen words
    all_text = ' '.join(df.occ1.tolist())
    unique_characters = set(all_text)
    
    new_tokenizer = tokenizer.train_new_from_iterator(
        list(df.occ1), 
        vocab_size=len(unique_characters), 
        initial_alphabet=unique_characters)
    return(tokenizer)

# %%
# Build the Sentiment Classifier class 
class LSTMOccupationClassifier(nn.Module):
    
    # Constructor class 
    def __init__(self, n_classes, model_domain, tokenizer, dropout_rate):
        super(BERTOccupationClassifier, self).__init__()
        self.bert = bert_model(model_domain, tokenizer)
        self.drop = nn.Dropout(p=dropout_rate)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
    
    # Forward propagaion class
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
          input_ids=input_ids,
          attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output
        
        #  Add a dropout layer 
        output = self.drop(pooled_output)
        return self.out(output)