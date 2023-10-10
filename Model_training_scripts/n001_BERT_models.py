# -*- coding: utf-8 -*-
"""
Models
Created on Tue May 23 15:02:16 2023

Authors: Christian Vedel, [christian-vs@sam.sdu.dk],

Purpose: Defines model classes
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

# %% BERT finetune based on model_domain
def BERT_path(model_domain):
    if(model_domain == "DK_CENSUS"):
        MDL = 'Maltehb/danish-bert-botxo' # https://huggingface.co/Maltehb/danish-bert-botxo
    elif(model_domain == "EN_MARR_CERT"):
        MDL = "bert-base-uncased" 
    elif(model_domain == "HSN_DATABASE"):
        MDL = "GroNLP/bert-base-dutch-cased"
    else:
        raise Exception("This is not implemented yet")
        
    return MDL

#%% Tokenizer
def load_tokenizer(model_domain):
    MDL = BERT_path(model_domain)
    tokenizer = BertTokenizer.from_pretrained(MDL)
    return(tokenizer)

#%% Update tokenizer
def update_tokenizer(tokenizer, df):
    # Add unseen words
    all_text = ' '.join(df.occ1.tolist())
    words_list = all_text.split()
    unique_words = set(words_list)
    # Add tokens for missing words
    tokenizer.add_tokens(list(unique_words))
    
    return tokenizer

# %%
def bert_model(model_domain, tokenizer):
    MDL = BERT_path(model_domain)
    # Load the basic BERT model 
    bert_model = BertModel.from_pretrained(MDL)
    
    # Adapt model size to the tokens added:
    #resize_token_embeddings method and set it to a multiple of 8, which is the number of 
    #Tensor Cores in an NVIDIA A40 GPU.
    bert_model.resize_token_embeddings(len(tokenizer),pad_to_multiple_of=48) 
    
    return bert_model

# %%
# Build the Sentiment Classifier class 
class BERTOccupationClassifier(nn.Module):
    
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