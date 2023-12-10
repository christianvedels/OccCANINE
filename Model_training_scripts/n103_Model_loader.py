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
from n001_Model_assets import * 
from n102_DataLoader import Concat_string, Concat_string_canine

from transformers import AutoTokenizer
from unidecode import unidecode
import numpy as np

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
    
# Get_adapated_tokenizer("CANINE")

# %% Load best model instance
# Define the path to the saved binary file
class Finetuned_model:
    def __init__(self, name, device = "cpu"):
        
        self.name = name
        self.device = device
        
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
        model.load_state_dict(loaded_state)
        model.to(device)   
        
        self.model = model
        
    def encode(self, occ1, lang, concat_in):
        if not concat_in: # Because then it is assumed that strings are already clean
            occ1 = [occ.lower() for occ in occ1]
            occ1 = [unidecode(occ) for occ in occ1]
        
        # Handle singular lang
        if isinstance(lang, str):
            lang = [lang]
        
        # Make list
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
            
    def predict(self, occ1, lang = "unk", batch_size = 256, what = "logits", threshold = 0.5, concat_in = False, verbose = False, ):
        """
        occ1:           List of occupational strings
        lang:           Language (defaults to unknown)
        batch_size:     How to batch up the data
        what:           What to return "logits", "probs", "pred", "bin" [n] return top [n]
        threshold:      Prediction threshold in case of what == "pred"
        verbose:        Should updates be printed?  
        concat_in:      Is the input already concated? E.g. [occ1][SEP][lang]
        """
        inputs = self.encode(occ1, lang, concat_in)
        
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
                print(f"Processed batch {batch_num} out of {total_batches} batches")
            
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
                
        # Convert to array if what == "bin"
        if what == "bin":
            results = np.array(results)
            
        return results
    
    def forward_base(self, occ1, lang, concat_in = False, verbose = False):
        """
        This method prints returns the forward pass of the underlying transformer model
        
        occ1:           List of occupational strings
        lang:           Language (defaults to unknown)
        concat_in:      Is the input already concated? E.g. [occ1][SEP][lang]
        verbose:        Should updates be printed?  
        """
        raise Exception("Not implemented yet")
        # Clean string
        # breakpoint()
        if not concat_in: # Because then it is assumed that strings are already clean
            occ1 = [occ.lower() for occ in occ1]
            occ1 = [unidecode(occ) for occ in occ1]
        
        # Define input
        if concat_in:
            inputs = occ1
        else:
            if "RoBERTa" in self.name:
                inputs = [Concat_string(occ, l) for occ, l in zip(occ1, lang)]
            elif "CANINE" in self.name:
                inputs = [Concat_string_canine(occ, l) for occ, l in zip(occ1, lang)]
                
        # Tokenize the batch of inputs
        tokenized_inputs = self.tokenizer(inputs, padding=True, truncation=True, return_tensors='pt')

        input_ids = tokenized_inputs['input_ids'].to(self.device)
        attention_mask = tokenized_inputs['attention_mask'].to(self.device)

        with torch.no_grad():
            output = self.model.basemodel(input_ids, input_ids)
                
        return output
    
    def test(self, y, yhat):
        test = []
        for p, y in zip(pred, df_bin):
            result = p == y
            test.append(all(result))
            
        np.mean(test)
                
        
        
        
    
# %% Test it
# name = "231115_CANINE_Multilingual_CANINE_sample_size_6_lr_2e-05_batch_size_32"

# Get_finetuned_model(name)
