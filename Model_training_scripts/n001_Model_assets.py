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
        
# %%
# class StackingEnsemble(nn.Module):
#     def __init__(self, base_models, meta_classifier, encoders, tokenizers, freeze_base_models=True):
#         super(StackingEnsemble, self).__init__()
        
#         # Create a list to hold the base models
#         self.base_models = nn.ModuleList(base_models)
        
#         # Freeze or unfreeze base models based on the argument
#         for model in self.base_models:
#             self.freeze_layers(model, freeze_base_models)
        
#         self.meta_classifier = meta_classifier
        
#         self.encoders = encoders
#         self.tokenizers = tokenizers

#     def freeze_layers(self, model, freeze):
#         if freeze:
#             for param in model.parameters():
#                 param.requires_grad = False
#         return model

#     def forward(self, occ1, lang):
#         # Make predictions with each base model using corresponding inputs
#         # [e(occ1, lang) for e in ] # Include concat function
        
#         predictions = [
#             base_model(input_ids, attention_mask) for (input_ids, attention_mask), base_model in zip(inputs, self.base_models)
#         ]
        
#         # Concatenate predictions along the feature dimension
#         features = torch.cat(predictions, dim=1)
        
#         # Forward pass through the meta-classifier using concatenated features
#         final_pred = self.meta_classifier(features)
        
#         return final_pred
    
