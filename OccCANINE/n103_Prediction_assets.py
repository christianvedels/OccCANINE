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
from OccCANINE.n001_Model_assets import CANINEOccupationClassifier, CANINEOccupationClassifier_hub, load_tokenizer
from OccCANINE.n102_DataLoader import Concat_string, Concat_string_canine, OCCDataset, read_data, labels_to_bin, TrainTestVal, save_tmp, create_data_loader
from OccCANINE.n101_Trainer import trainer_loop_simple, eval_model
from OccCANINE.n100_Attacker import AttackerClass

from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from unidecode import unidecode
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
import pandas as pd
import time

# %% Get_adapted_tokenizer
def Get_adapated_tokenizer(name):
    """
    This function loads the adapted tokenizers used in training
    """
    if "CANINE" in name:
        tokenizer = load_tokenizer("Multilingual_CANINE")
    else:
       tokenizer_save_path = name + '_tokenizer'
       tokenizer = AutoTokenizer.from_pretrained(tokenizer_save_path) 
       
    return tokenizer
    

# %% Top_n_to_df

def Top_n_to_df(result, top_n):
    """
    Converts dictionary of top n predictions to df
    Parameters
    ----------
    result:     List of dicitonaries from predict method in OccCANINE
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

# %% Class to load model

class OccCANINE:
    def __init__(self, name = "CANINE", device = None, batch_size = 256, verbose = False, baseline = False, hf = True, force_download = False):
        """
        Initializes the OccCANINE model with specified configurations.
        
        Parameters:
        - name (str): Name of the model to load. Defaults to "CANINE". For local models, specify the model name as it appears in 'Model'.
        - device (str or None): The computing device (CPU or CUDA) on which the model will run. If None, the device is auto-detected.
        - batch_size (int): The number of examples to process in each batch. This affects memory usage and processing speed.
        - verbose (bool): If True, progress updates and diagnostic information will be printed during model operations.
        - baseline (bool): If True, loads a baseline (untrained) version of CANINE. Useful for comparison studies.
        - hf (bool): If True, attempts to load the model from Hugging Face's model repository. If False, loads a local model specified by the 'name' parameter.
        - force_download (bool): If True, forces a re-download of the model from the Hugging Face's model repository even if it is already cached locally.
        
        Raises:
        - Exception: If 'hf' is False and a local model 'name' is not provided.
        """
        
        # Detect device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)
        if verbose:
            print(f"Using device: {self.device}")
        
        if name == "CANINE" and not hf:
            raise ValueError("When 'hf' is False, a specific local model 'name' must be provided.")
        
        self.name = name
        self.batch_size = batch_size
        self.verbose = verbose
        
        # Get tokenizer
        self.tokenizer = Get_adapated_tokenizer("CANINE_Multilingual_CANINE_sample_size_10_lr_2e-05_batch_size_256")
        
        # Get key
        self.key, self.key_desc = self._load_keys()
                
        self.model = self._load_model(hf, force_download, baseline)
        
    def _load_keys(self):
        # Load and return both the key and key with descriptions
        key_df = pd.read_csv("Data/Key.csv").iloc[1:]
        key = dict(zip(key_df.code, key_df.hisco))
        key_desc = dict(zip(key_df.code, key_df.en_hisco_text))
        return key, key_desc
    
    def _load_model(self, hf, force_download, baseline):
        if hf:
            return CANINEOccupationClassifier_hub.from_pretrained("christianvedel/OccCANINE", force_download=force_download).to(self.device)
        else:
            model_path = f'{self.name}.bin'
            loaded_state = torch.load(model_path, map_location=self.device)
            model = CANINEOccupationClassifier(model_domain="Multilingual_CANINE", n_classes=len(self.key), dropout_rate=0)
            if not baseline:
                model.load_state_dict(loaded_state)
            return model.to(self.device)
    
    def _encode(self, occ1, lang, concat_in):
        """
        Encodes occupational strings into a format suitable for model input.
        
        Parameters:
        - occ1 (list of str): A list of occupational strings to encode.
        - lang (str or list of str): The language(s) of the occupational strings. Can be a single language string or a list of language strings.
        - concat_in (bool): If True, assumes that the input strings are already concatenated in the required format (e.g., occupation[SEP]language). If False, performs concatenation.
        
        Returns:
        - list of str: The encoded inputs ready for model processing.
        """
        
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
            inputs = [Concat_string_canine(occ, l) for occ, l in zip(occ1, lang)]
                
        return inputs
            
    def predict(self, occ1, lang = "unk", what = "pred", threshold = 0.22, concat_in = False, get_dict = False, get_df = True):
        """
        Makes predictions on a batch of occupational strings.
        
        Parameters:
        - occ1 (list of str): A list of occupational strings to predict.
        - lang (str, optional): The language of the occupational strings. Defaults to "unk" (unknown).
        - what (str or int, optional): Specifies what to return. Options are "logits", "probs", "pred", "bin", or an integer [n] to return top [n] predictions. Defaults to "pred".
        - threshold (float, optional): The prediction threshold for binary classification tasks. Defaults to 0.22. Which is generally optimal for F1.
        - concat_in (bool, optional): Specifies if the input is already concatenated in the format [occupation][SEP][language]. Defaults to False.
        - get_dict (bool, optional): If True and 'what' is an integer, returns a list of dictionaries with predictions instead of a DataFrame. Defaults to False.
        - get_df (bool, optional): If True and 'what' equals "pred", returns predictions in a DataFrame format. Defaults to True.
        
        Returns:
        - Depends on the 'what' parameter. Can be logits, probabilities, predictions, a binary matrix, or a DataFrame containing the predicted classes and probabilities.
        """
        
        # Setup
        inputs = self._encode(occ1, lang, concat_in)
        batch_size = self.batch_size
        verbose = self.verbose
        results = []
        total_batches = (len(inputs) + batch_size - 1) // batch_size  # Calculate the total number of batches
        
        # Timing
        start = time.time()
        
        # Fix get dict conditionally
        if get_dict:
            get_df = False
        
        if get_df:
            what0 = what
            what = 5 # This is the easiest way of handling this

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
            elif what == "tokens":
                results.append(batch_input_ids)
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
                
        if isinstance(what, (int, float)) and what0 == "pred":
            # Print update
            print("\nPrediction done. Cleaning results.")
            # Disable preds for all below threshold
            for j in range(1, what+1):
                probs_j = results[f"prob_{j}"]
                test_j = probs_j > threshold
                for i in range(results.shape[0]):
                    if not test_j[i]:
                        results.loc[i,f"hisco_{j}"] = float("NaN")
                        results.loc[i, f"desc_{j}"] = "No pred"
                        results.loc[i, f"prob_{j}"] = float("NaN")
            
            results.insert(0,'inputs', inputs)
            results["hisco_1"] = results["hisco_1"].fillna("-1")
                
        end = time.time()
        
        if verbose:
            dif_time = end - start
            m, s = divmod(dif_time, 60)
            h, m = divmod(m, 60)
            
            print(f"Produced HISCO codes for {occ1.shape[0]} observations in {h:.0f} hours, {m:.0f} minutes and {s:.3f} seconds.")
            
            saved_time = occ1.shape[0]*10 - dif_time
            m, s = divmod(saved_time, 60)
            h, m = divmod(m, 60)
            
            print("Estimated hours saved compared to human labeller (assuming 10 seconds per label):")
            print(f" ---> {h:.0f} hours, {m:.0f} minutes and {s:.0f} seconds")
                
        return results
    
    def forward_base(self, occ1, lang = "unk", concat_in = False):
        """
        This method prints returns the forward pass of the underlying transformer model
        
        occ1:           List of occupational strings
        lang:           Language (defaults to unknown)
        concat_in:      Is the input already concated? E.g. [occ1][SEP][lang]
        """
        inputs = self._encode(occ1, lang, concat_in)
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
        return results
    
    def _process_data(self, data_df, label_cols, batch_size, model_domain = "Multilingual_CANINE", alt_prob = 0.2, insert_words = True, testval_fraction = 0.1, new_labels = True, verbose = False):
        """
        Processes the input data for training or validation.
    
        This internal method prepares the data for the model by converting labels to numeric codes, merging with a key DataFrame, handling missing values, tokenizing the text, and creating data loaders for training and validation.
    
        Parameters
        ----------
        data_df : pd.DataFrame
            The input data containing text and labels.
        label_cols : list of str
            The column names in data_df that contain the labels.
        batch_size : int
            The size of the batch to be used in data loaders.
        model_domain : str, optional
            The domain of the model to be used. Defaults to "Multilingual_CANINE".
        alt_prob : float, optional
            The probability of using alternative tokens (data augmentation). Defaults to 0.2.
        insert_words : bool, optional
            Whether to insert words as a part of data augmentation. Defaults to True.
        testval_fraction : float, optional
            The fraction of data to be used for validation. Defaults to 0.1.
        new_labels : bool, optional
            If True, the method generates a new key based on the unique labels in the input data and updates data processing accordingly. Defaults to False.
        verbose : bool, optional
            Whether to print out the training progress. Defaults to True.
    
        Returns
        -------
        dict
            A dictionary containing training and validation data loaders, and the tokenizer.
        """
        if new_labels:
            # Generate a new key based on the unique labels in the label columns
            unique_labels = pd.unique(data_df[label_cols].values.ravel('K'))
            key = {label: idx for idx, label in enumerate(unique_labels)}
            self.key = key
            
            # Convert the new key dictionary to a DataFrame for joining
            key_df = pd.DataFrame(list(self.key.items()), columns=['Hisco', 'Code']) # Yes it is not HISCO, but if we just call it that everything runs
            
            if verbose:
                print(f"Produced new key for {len(unique_labels)} possible labels")
        else:
            breakpoint()
            key = self.key
            # Convert the key dictionary to a DataFrame for joining
            key_df = pd.DataFrame(list(key.items()), columns=['Code', 'Hisco']) 
            # Convert 'Hisco' in key_df to numeric (float), as it's going to be merged with numeric columns
            key_df['Hisco'] = pd.to_numeric(key_df['Hisco'], errors='coerce')
            # Ensure the label columns are numeric and have the same type as the key
            for col in label_cols:
                data_df[col] = pd.to_numeric(data_df[col], errors='coerce')
        
        # Define expected code column names
        expected_code_cols = [f'code{i}' for i in range(1, 6)]
        
        # Drop code1 to code5 from data_df if they exist
        data_df = data_df.drop(columns=expected_code_cols, errors='ignore')
        
        # Initialize an empty DataFrame to store the label codes
        label_codes = pd.DataFrame(index=data_df.index)
        
        # Iterate over each label column and perform left join with the key
        for i, col in enumerate(label_cols, start=1):
            # Perform the left join
            merged_df = pd.merge(data_df[[col]], key_df, left_on=col, right_on='Hisco', how='left')
        
            # Assign the 'Code' column from the merged DataFrame to label_codes
            label_codes[f'code{i}'] = merged_df['Code']
        
        # Ensure label_codes has columns code1 to code5, filling missing ones with NaN
        label_codes = label_codes.reindex(columns=expected_code_cols, fill_value=np.nan)
        
        # Handle missing values by ensuring they are NaN
        label_codes = label_codes.apply(pd.to_numeric, errors='coerce')
        
        # Concatenate label_codes with data_df
        data_df = pd.concat([data_df, label_codes], axis=1)
         
        try:
            _ = labels_to_bin(label_codes, max_value = np.max(key_df["Code"]+1))
        except Exception:
            raise Exception("Was not able to convert to binary representation")
        
        # Tokenizer
        tokenizer = self.tokenizer
        
        # Define attakcer instance
        attacker = AttackerClass(data_df)
        
        # Split data
        df_train, df_val = TrainTestVal(data_df, verbose=False, testval_fraction = testval_fraction, test_size = 0)
        
        # To use later
        n_obs_train = df_train.shape[0]
        n_obs_val = df_val.shape[0]
        
        # Print split sizes
        print(f"{n_obs_train} observations will be used in training.")
        print(f"{n_obs_val} observations will be used in validation.")
        
        # Save tmp files
        save_tmp(df_train, df_val, df_test = df_val, path = "../Data/Tmp_finetune")
        
        # File paths for the index files
        train_index_path = "../Data/Tmp_finetune/Train_index.txt"
        val_index_path = "../Data/Tmp_finetune/Val_index.txt"
        
        # Calculate number of classes
        n_classes = len(key)

        # Instantiating OCCDataset with index file paths
        ds_train = OCCDataset(df_path="../Data/Tmp_finetune/Train.csv", n_obs=n_obs_train, tokenizer=tokenizer, attacker=attacker, max_len=128, n_classes=n_classes, index_file_path=train_index_path, alt_prob=0, insert_words=False, model_domain=model_domain, unk_lang_prob = 0)
        ds_train_attack = OCCDataset(df_path="../Data/Tmp_finetune/Train.csv", n_obs=n_obs_train, tokenizer=tokenizer, attacker=attacker, max_len=128, n_classes=n_classes, index_file_path=train_index_path, alt_prob=alt_prob, insert_words=insert_words, model_domain=model_domain, unk_lang_prob = 0)
        ds_val = OCCDataset(df_path="../Data/Tmp_finetune/Val.csv", n_obs=n_obs_val, tokenizer=tokenizer, attacker=attacker, max_len=128, n_classes=n_classes, index_file_path=val_index_path, alt_prob=0, insert_words=False, model_domain=model_domain, unk_lang_prob = 0)
        
        # Data loaders
        data_loader_train, data_loader_train_attack, data_loader_val, _ = create_data_loader(
            ds_train, ds_train_attack, ds_val, ds_val, # DS val twice as dummy to make it run
            batch_size = batch_size
            )
        
        return {
            'data_loader_train': data_loader_train,
            'data_loader_train_attack': data_loader_train_attack,
            'data_loader_val': data_loader_val,
            'tokenizer': self.tokenizer
        }
        
    
    def _train_model(self, processed_data, model_name, epochs, only_train_final_layer, verbose = True, verbose_extra = False, new_labels = False):
        """
        Trains the model with the provided processed data.
    
        This internal method sets up the optimizer, scheduler, and loss function, and initiates the training loop. It also handles layer freezing for transfer learning, if specified.
    
        Parameters
        ----------
        processed_data : dict
            The dictionary containing training and validation data loaders, and the tokenizer.
        model_name : str
            The name of the model to be used for saving.
        epochs : int
            The number of epochs for training.
        only_train_final_layer : bool
            Whether to train only the final layer of the model.
        verbose : bool, optional
            Whether to print out the training progress. Defaults to True.
        verbose_extra : bool, optional
            Whether to print extra details during training. Defaults to False.
        new_labels : bool, optional
            Whether new labels should be used. Defaults to false
    
        Returns
        -------
        tuple
            A tuple containing the training history and the trained model.
        """
        
        if new_labels:
            n_classes = len(self.key)
            self.model.out = nn.Linear(in_features=self.model.out.in_features, out_features=n_classes)
            
            # Ensure the model is set to the correct device again
            self.model = self.model.to(self.device)
        
        optimizer = AdamW(self.model.parameters(), lr=2*10**-5)
        total_steps = len(processed_data['data_loader_train']) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # Set the loss function 
        loss_fn = nn.BCEWithLogitsLoss().to(self.device)
        
        # Freeze layers
        if only_train_final_layer:
            # Freeze all layers initially
            for param in self.model.parameters():
                param.requires_grad = False
     
            # Unfreeze the final layers (self.out in CANINEOccupationClassifier)
            for param in self.model.out.parameters():
                param.requires_grad = True
                
                
        val_acc, val_loss = eval_model(
            self.model,
            processed_data['data_loader_val'],
            loss_fn = loss_fn,
            device = self.device,
            )
        
        print("----------")
        if verbose:
            print(f"Intital performance:\nValidation acc: {val_acc}; Validation loss: {val_loss}")
           
        history, model = trainer_loop_simple(
            self.model,
            epochs = epochs, 
            model_name = model_name,
            data = processed_data,
            loss_fn = loss_fn,
            optimizer = optimizer,
            device = self.device,
            scheduler = scheduler,
            verbose = verbose,
            verbose_extra  = verbose_extra,
            attack_switch = True,
            initial_loss = val_loss
            )
        
        # == Load best model ==
        # breakpoint()
        # Load state
        model_path = '../OccCANINE/Finetuned/'+model_name+'.bin'
        if not os.path.isfile(model_path):
            print("Model did not improve in training. Realoding original model")
            model_path = self.name+'.bin'
   
        # Load the model state
        loaded_state = torch.load(model_path)
        
        # If-lookup for model
        config = {
            "model_domain": "Multilingual_CANINE",
            "n_classes": len(self.key),
            "dropout_rate": 0,
            "model_type": "canine"
        }
        
        model = CANINEOccupationClassifier_hub(config)
        
        model.load_state_dict(loaded_state)        
        model.to(self.device)   
        self.model = model
        
        print("Loaded best version of model")
        
        val_acc, val_loss = eval_model(
            self.model,
            processed_data['data_loader_val'],
            loss_fn = loss_fn,
            device = self.device,
            )
        
        print("----------")
        if verbose:
            print(f"Final performance:\nValidation acc: {val_acc}; Validation loss: {val_loss}")
            
        return history, model
                
    def finetune(
            self, 
            data_df, 
            label_cols, 
            batch_size="Default", 
            epochs=3, 
            attack=True, 
            save_name = "finetuneCANINE", 
            only_train_final_layer = False, 
            verbose = True,
            verbose_extra = False, 
            test_fraction = 0.1,
            new_labels = False
            ):
        """
        Fine-tunes the model on the provided dataset.
    
        This method orchestrates the fine-tuning process by processing the data, training the model, and handling the batch size specifications. It also allows for data augmentation and control over verbosity during training.
    
        Parameters
        ----------
        data_df : pd.DataFrame
            DataFrame containing the training data. Must include:
                - 'occ1', the occupational description
                - columns with labels specified in 'label_cols'
                - 'lang', a column of the language of each label
        label_cols : list of str
            Names of the columns in data_df that contain the labels.
        batch_size : int or "Default", optional
            Batch size for training. If "Default", the class-specified batch size is used. Defaults to "Default".
        epochs : int, optional
            Number of epochs for training. Defaults to 3.
        attack : bool, optional
            Indicates whether data augmentation should be used in training. Defaults to True.
        save_name : str, optional
            The name under which the trained model is to be saved. Defaults to "finetuneCANINE".
        only_train_final_layer : bool, optional
            Whether to train only the final layer of the model. Defaults to False.
        verbose : bool, optional 
            Should updates be printed. Defaults to True
        verbose_extra : bool, optional
            Whether to print additional information during training. Defaults to False.
        test_fraction : float, optional
            The fraction of the dataset to be used for testing. Defaults to 0.1.
        new_labels : bool, optional
            Whether the labels are a new system of labeling. Defaults to False.
    
        Returns
        -------
        None
        """
        
        print("======================================")
        print("==== Started finetuning procedure ====")
        print("======================================")
        
        # Handle batch_size
        if batch_size=="Default":
            batch_size = self.batch_size
        
        # Load and process the data
        processed_data = self._process_data(
            data_df, label_cols, batch_size=batch_size,
            testval_fraction = test_fraction,
            new_labels = new_labels,
            verbose = verbose
            )

        # Training the model
        self._train_model(
            processed_data, model_name = save_name, epochs = epochs, only_train_final_layer=only_train_final_layer,
            verbose_extra = verbose_extra,
            new_labels = new_labels
            )

        print("Finetuning completed successfully.")
        
