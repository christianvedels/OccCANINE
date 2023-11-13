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
import transformers
from transformers import BertModel, BertTokenizer, XLMRobertaTokenizer, XLMRobertaModel, CanineTokenizer, CanineModel
import torch
import torch.nn as nn
from keras.layers import TextVectorization

# %% BERT finetune based on model_domain
def modelPath(model_domain):
    if(model_domain == "DK_CENSUS"):
        MDL = 'Maltehb/danish-bert-botxo' # https://huggingface.co/Maltehb/danish-bert-botxo
    elif(model_domain == "EN_MARR_CERT"):
        MDL = "bert-base-uncased" 
    elif(model_domain == "HSN_DATABASE"):
        MDL = "GroNLP/bert-base-dutch-cased"
    elif(model_domain == "Multilingual"):
        MDL = 'xlm-roberta-base'
    elif model_domain == "Multilingual_CANINE":  # Replace "CANINE_MODEL_NAME" with the actual CANINE model name
        MDL = "google/canine-s"          
    else:
        raise Exception("This is not implemented yet")
        
    return MDL

#%% Tokenizer
def load_tokenizer(model_domain):
    # breakpoint()
    MDL = modelPath(model_domain)
    if MDL == "xlm-roberta-base":
        tokenizer = XLMRobertaTokenizer.from_pretrained(MDL)
    elif MDL == "google/canine-s":  # Replace "CANINE_MODEL_NAME" with the actual CANINE model name
        tokenizer = CanineTokenizer.from_pretrained(MDL)
    else: 
        raise Exception("Not implemented")
        
        # Consider implementing this
        tokenizer = BertTokenizer.from_pretrained(MDL)
    return(tokenizer)

# load_tokenizer(model_domain="Multilingual")

#%% Update tokenizer
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

# %%
def getModel(model_domain, tokenizer):
    # breakpoint()
    MDL = modelPath(model_domain)
    # Load the basic BERT model 
    if model_domain == "Multilingual":
        model = XLMRobertaModel.from_pretrained(MDL)
    elif model_domain == "Multilingual_CANINE":  
        model = CanineModel.from_pretrained(MDL)
    else:
        raise Exception("Not implemented")
        # model = BertModel.from_pretrained(MDL)
    
    if model_domain == "Multilingual_CANINE": # No reason to resize these tokens
        return model
    
    # Adapt model size to the tokens added:   
    model.resize_token_embeddings(len(tokenizer))
    
    return model

# %%
# Build the Sentiment Classifier class 
class BERTOccupationClassifier(nn.Module):
    
    # Constructor class 
    def __init__(self, n_classes, model_domain, tokenizer, dropout_rate):
        super(BERTOccupationClassifier, self).__init__()
        self.basemodel = getModel(model_domain, tokenizer)
        self.drop = nn.Dropout(p=dropout_rate)
        self.out = nn.Linear(self.basemodel.config.hidden_size, n_classes)
    
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
    
# %%
# Build the Classifier 
class XMLRoBERTaOccupationClassifier(nn.Module):
    
    # Constructor class 
    def __init__(self, n_classes, model_domain, tokenizer, dropout_rate):
        super(XMLRoBERTaOccupationClassifier, self).__init__()
        self.basemodel = getModel(model_domain, tokenizer)
        self.drop = nn.Dropout(p=dropout_rate)
        self.out = nn.Linear(self.basemodel.config.hidden_size, n_classes)
    
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
    
# %%
# Build the Classifier 
class CANINEOccupationClassifier(nn.Module):
    
    # Constructor class 
    def __init__(self, n_classes, model_domain, tokenizer, dropout_rate):
        super(CANINEOccupationClassifier, self).__init__()
        self.basemodel = getModel(model_domain, tokenizer)
        self.drop = nn.Dropout(p=dropout_rate)
        self.out = nn.Linear(self.basemodel.config.hidden_size, n_classes)
    
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
    
# %% LSTM
class CharLSTMOccupationClassifier(nn.Module):
    def __init__(self, n_classes, unique_chars, unique_langs, char_embedding_size, lang_embedding_size, hidden_size, num_layers, dropout_rate, char_length):
        super(CharLSTMOccupationClassifier, self).__init__()
        
        # breakpoint()
        # Makeshift lang tokenizer
        # Assume you have the following mappings from the tokenizer
        self.char_to_id_mapping = {char: idx for idx, char in enumerate([' '] + unique_chars)}
        self.lang_to_id_mapping = {lang: idx for idx, lang in enumerate(unique_langs)}
        self.char_length = char_length
        
        self.char_embedding = nn.Embedding(len(self.char_to_id_mapping), char_embedding_size)
        self.lang_embedding = nn.Embedding(len(self.lang_to_id_mapping), lang_embedding_size)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.lstm = nn.LSTM(input_size=char_embedding_size + lang_embedding_size,
                            hidden_size=hidden_size, num_layers=num_layers,
                            bidirectional=True, batch_first=True)
        
        self.global_avg_pooling = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.output_layer = nn.Linear(hidden_size, n_classes)
        self.leaky_relu = nn.LeakyReLU(0.01)
                
    # Function to convert characters to ids
    def char_to_ids(self, char_sequence):
        char_ids = [self.char_to_id_mapping[char] for char in char_sequence]
        
        # Padding to char_length
        padding_length = self.char_length - len(char_ids)
        padded_char_ids = char_ids + [0] * padding_length if padding_length > 0 else char_ids
        
        return torch.tensor(padded_char_ids)
    
    # Function to convert languages to ids
    def lang_to_ids(self, lang_token):
        res = self.lang_to_id_mapping[lang_token]
        return torch.tensor(res)

    def forward(self, occ_input, lang_input):
        # breakpoint()
        
        # Convert lang to unk if unknown language
        lang_input = lang_input.lower()
        # Check if lang_input is in self.lang_to_id_mapping or convert to unknown
        if lang_input not in self.lang_to_id_mapping:
            lang_input = 'unk'
        
        # Convert tokens to IDs
        char_ids = self.char_to_ids(occ_input)

        # Convert language token to ID
        lang_ids = self.lang_to_ids(lang_input)
        
        char_embedded = self.char_embedding(char_ids)
        lang_embedded = self.lang_embedding(lang_ids)
        
        # Concatenate character and language embeddings along the last dimension
        # breakpoint()
        concatenated_input = torch.cat((lang_embedded, char_embedded), dim=-1)

        dropped = self.dropout(concatenated_input)
        lstm_out, _ = self.lstm(dropped)
        
        # Global average pooling
        pooled_output = self.global_avg_pooling(lstm_out.permute(0, 2, 1)).squeeze(-1)
        
        fc1_out = self.leaky_relu(self.fc1(pooled_output))
        fc2_out = self.leaky_relu(self.fc2(fc1_out))
        
        output = torch.sigmoid(self.output_layer(fc2_out))
        
        return output
    


