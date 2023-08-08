# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 13:10:35 2023

@author: christian-vs
"""

# %% Key vars
model_domain = "DK_CENSUS"
model_name = "XMLRoBERTa_DK_CENSUS_sample_size4"
MAX_LEN = 200
BATCH_SIZE = 8
toyrun = True

# %% Import packages
import pandas as pd
import numpy as np
import torch
import os
import random as r
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerFast

# %% Check cuda availability
is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

device = torch.device("cpu")    

#%% Load data
if os.environ['COMPUTERNAME'] == 'SAM-126260':
    os.chdir('D:/Dropbox/PhD/HISCO clean')
else:
    os.chdir('C:/Users/chris/Dropbox/PhD/HISCO clean')
print(os.getcwd())

if(model_domain == "DK_CENSUS"):
    fname = "Data/Training_data/DK_census_train.csv"
else:
    raise Exception("This is not implemented yet")

df = pd.read_csv(fname, encoding = "UTF-8")

key = pd.read_csv("Data/Key.csv") # Load key and convert to dictionary
key = key[1:]
key = zip(key.code, key.hisco)
key = list(key)
key = dict(key)
label2id = key
id2label = {v: k for k, v in label2id.items()}

# Downsample for toyrun
if toyrun:
    r.seed(20)
    df = df.sample(100)

# %% Construct prediction function
def predict_hiscos(strings):
    x = tokenizer(
        strings,
        max_length=MAX_LEN,
        pad_to_max_length=True,
        return_tensors="pt"
    )
    ids = x["input_ids"]
    mask = x["attention_mask"]
    
    # Forward pass through the model
    logits = model(ids, mask)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(logits.squeeze().cpu())
    
    # Sort probabilities and get the top 5 indices
    top_probs, top_indices = torch.topk(probs, k=5, dim=1)
    
    # Convert top indices to predicted labels
    predicted_labels = []
    for obs_indices in top_indices:
        obs_labels = [key[i] for i in obs_indices.numpy()]
        predicted_labels.append(obs_labels)
    
    # Create hiscos_df DataFrame
    hiscos_df = pd.DataFrame(predicted_labels, columns=['hisco1', 'hisco2', 'hisco3', 'hisco4', 'hisco5'])
        
    # Create hisco_prob_df DataFrame
    hisco_prob_df = pd.DataFrame(top_probs.detach().numpy(), columns=['prob_hisco1', 'prob_hisco2', 'prob_hisco3', 'prob_hisco4', 'prob_hisco5'])
    
    return hiscos_df, hisco_prob_df

# %% Tokenizer and model
# Define the paths for loading the model and tokenizer
path = "Trained_models/" + model_name
path_vocab = path + "/tokenizer"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(path_vocab)

# Load the model
output_model_file = path + "/model.bin"
model = torch.load(output_model_file)

# %% Test the loaded model and tokenizer
text = ["han er fisker", "han er bonde"]
predict_hiscos(text)

