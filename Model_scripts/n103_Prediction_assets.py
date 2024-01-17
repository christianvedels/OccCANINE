# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 23:03:44 2023
Loads trained version of the models
@author: chris
"""

import os
script_directory = os.path.dirname(os.path.abspath(__name__))
os.chdir(script_directory)

# %% Import modules
from n001_Model_assets import XMLRoBERTaOccupationClassifier, CANINEOccupationClassifier, load_tokenizer
from n102_DataLoader import Concat_string, Concat_string_canine

from transformers import AutoTokenizer
from unidecode import unidecode
import numpy as np
import torch
import pandas as pd

# %% Get_adapted_tokenizer
def Get_adapated_tokenizer(name):
    """
    This function loads the adapted tokenizers used in training
    """
    if "CANINE" in name:
        tokenizer = load_tokenizer("Multilingual_CANINE")
    else:
       tokenizer_save_path = '../Trained_models/' + name + '_tokenizer'
       tokenizer = AutoTokenizer.from_pretrained(tokenizer_save_path) 
       
    return tokenizer
    

# %% Top_n_to_df

def Top_n_to_df(result, top_n):
    """
    Converts dictionary of top n predictions to df
    Parameters
    ----------
    result:     List of dicitonaries from predict method in Finetuned_model
    top_n:      Number of predictions

    Returns:    pd.DataFrame
    -------

    """
    
    data = result

    rows = []
    for d in data:
        row = []
        for i in range(top_n):  # Assuming that each dictionary has keys '0' to '9'
            # Append each of the three elements in the tuple to the row list
            row.extend(d[i])
        rows.append(row)

    # Define the column names
    column_names = []
    for i in range(1, top_n+1):
        column_names.extend([f'hisco_{i}', f'prob_{i}', f'desc_{i}'])

    # Create a DataFrame
    x = pd.DataFrame(rows, columns=column_names)
    
    return(x)

# %% Load best model instance
# Define the path to the saved binary file
class Finetuned_model:
    def __init__(self, name, device = "cpu", batch_size = 256, verbose = False, baseline = False):
        """
        name:           Name of the model to load (name in 'Trained_models')
        device:         Which device should be used? Defaults to cpu
        batch_size:     How to batch up the data
        verbose:        Should updates be printed?  
        baseline:       Option to load baseline (untrained) version of the model
        """
        
        self.name = name
        self.device = device
        self.batch_size = batch_size
        self.verbose = verbose
        
        # Get tokenizer
        self.tokenizer = Get_adapated_tokenizer(name)
        
        # Load state
        model_path = '../Trained_models/'+name+'.bin'
   
        # Load the model state
        loaded_state = torch.load(model_path)
        
        # Load key
        key = pd.read_csv("../Data/Key.csv") # Load key and convert to dictionary
        key = key[1:]
        key = zip(key.code, key.hisco)
        key = list(key)
        self.key = dict(key)
        
        # Load key but with descriptions
        key = pd.read_csv("../Data/Key.csv") # Load key and convert to dictionary
        key = key[1:]
        key = zip(key.code, key.en_hisco_text)
        key = list(key)
        self.key_desc = dict(key)
        
        # If-lookup for model
        if "RoBERTa" in name:
            model = XMLRoBERTaOccupationClassifier(
                n_classes = len(self.key), 
                model_domain = "Multilingual", 
                tokenizer = self.tokenizer, 
                dropout_rate = 0
                )
        elif "CANINE" in name:
            model = CANINEOccupationClassifier(
                model_domain = "Multilingual_CANINE",
                n_classes = len(self.key), 
                tokenizer = self.tokenizer, 
                dropout_rate = 0
                )
        else:
            raise Exception(f"Was not able to identify/find {name}")
                
        # Update states and load onto device 
        if not baseline:
            model.load_state_dict(loaded_state)
        
        model.to(device)   
        
        self.model = model
        
    def encode(self, occ1, lang, concat_in):
        # breakpoint()
        if not concat_in: # Because then it is assumed that strings are already clean
            occ1 = [occ.lower() for occ in occ1]
            occ1 = [unidecode(occ) for occ in occ1]
        
        # Handle singular lang
        if isinstance(lang, str):
            lang = [lang]
            lang = [lang[0] for i in occ1]
                
        # Define input
        if concat_in:
            inputs = occ1
        else:
            if "RoBERTa" in self.name:
                inputs = [Concat_string(occ, l) for occ, l in zip(occ1, lang)]
            elif "CANINE" in self.name:
                inputs = [Concat_string_canine(occ, l) for occ, l in zip(occ1, lang)]
                
        return inputs
            
    def predict(self, occ1, lang = "unk", what = "logits", threshold = 0.5, concat_in = False, get_dict = False):
        """
        occ1:           List of occupational strings
        lang:           Language (defaults to unknown)
        batch_size:     How to batch up the data
        what:           What to return "logits", "probs", "pred", "bin" [n] return top [n]
        threshold:      Prediction threshold in case of what == "pred"
        concat_in:      Is the input already concated? E.g. [occ1][SEP][lang]
        get_dict:       For what [n] method this is an option to return a list of dictionaries
        """
        # breakpoint()
        inputs = self.encode(occ1, lang, concat_in)
        batch_size = self.batch_size
        verbose = self.verbose
        results = []
        total_batches = (len(inputs) + batch_size - 1) // batch_size  # Calculate the total number of batches

        for batch_num, i in enumerate(range(0, len(inputs), batch_size), 1):

            batch_inputs = inputs[i:i + batch_size]

            # Tokenize the batch of inputs
            batch_tokenized = self.tokenizer(batch_inputs, padding=True, truncation=True, return_tensors='pt')

            batch_input_ids = batch_tokenized['input_ids'].to(self.device)
            batch_attention_mask = batch_tokenized['attention_mask'].to(self.device)

            with torch.no_grad():
                output = self.model(batch_input_ids, batch_attention_mask)
            
            # === Housekeeping ====
            if batch_num % 1 == 0 and verbose:
                print(f"\rProcessed batch {batch_num} out of {total_batches} batches", end = "")
            
            # === Return what to return ====  
            # breakpoint()
            batch_logits = output
            if what == "logits":
                results.append(batch_logits)
            elif what == "probs":
                batch_predicted_probs = torch.sigmoid(batch_logits).cpu().numpy()
                results.append(batch_predicted_probs)
            elif what == "pred":
                batch_predicted_probs = torch.sigmoid(batch_logits).cpu().numpy()
                for probs in batch_predicted_probs:
                    labels = [[self.key[i], prob, self.key_desc[i]] for i, prob in enumerate(probs) if prob > threshold]
                    results.append(labels)
            elif isinstance(what, (int, float)):
                batch_predicted_probs = torch.sigmoid(batch_logits).cpu().numpy()
                for probs in batch_predicted_probs:
                    # Get the indices of the top 5 predictions
                    top5_indices = np.argsort(probs)[-what:][::-1]
            
                    # Create a list of tuples containing label and probability for the top 5 predictions
                    labels = [(self.key[i], probs[i], self.key_desc[i]) for i in top5_indices]
                    results.append(labels)
            elif what == "bin":
                # Return binary matrix with cols equal to codes
                batch_predicted_probs = torch.sigmoid(batch_logits).cpu().numpy()
                for probs in batch_predicted_probs:
                    bin_out = [1 if prob > threshold else 0 for prob in probs]
                    results.append(bin_out)                
            else:
                raise Exception("'what' incorrectly specified")
                
        # Clean results
        if what == "bin":
            results = np.array(results)
        
        if what == "probs":
            results = np.concatenate(results, axis=0)
            
        if isinstance(what, (int, float)):
            if not get_dict:
                results = Top_n_to_df(results, what)
                    
        print("\n")
        return results, inputs
    
    def forward_base(self, occ1, lang = "unk", concat_in = False):
        """
        This method prints returns the forward pass of the underlying transformer model
        
        occ1:           List of occupational strings
        lang:           Language (defaults to unknown)
        concat_in:      Is the input already concated? E.g. [occ1][SEP][lang]
        """
        inputs = self.encode(occ1, lang, concat_in)
        batch_size = self.batch_size
        verbose = self.verbose
        results = []
        total_batches = (len(inputs) + batch_size - 1) // batch_size  # Calculate the total number of batches

        for batch_num, i in enumerate(range(0, len(inputs), batch_size), 1):

            batch_inputs = inputs[i:i + batch_size]

            # Tokenize the batch of inputs
            batch_tokenized = self.tokenizer(batch_inputs, padding=True, truncation=True, return_tensors='pt')

            batch_input_ids = batch_tokenized['input_ids'].to(self.device)
            batch_attention_mask = batch_tokenized['attention_mask'].to(self.device)

            with torch.no_grad():
                res_i = self.model.basemodel(batch_input_ids, batch_attention_mask)
                
            results.append(res_i["pooler_output"])
            
            # === Housekeeping ====
            if batch_num % 1 == 0 and verbose:
                print(f"\rProcessed batch {batch_num} out of {total_batches} batches", end = "")
        
        results = torch.cat(results, axis=0).cpu().detach().numpy()
        
        print("\n")
        return(results)
