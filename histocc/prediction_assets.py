# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 23:03:44 2023
Loads trained version of the models
@author: chris
"""


import os
import time

from typing import Dict, Tuple

import torch

from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from unidecode import unidecode

import numpy as np
import pandas as pd

from .datasets import DATASETS
from .model_assets import (
    CANINEOccupationClassifier, 
    CANINEOccupationClassifier_hub, 
    Seq2SeqOccCANINE,
    Seq2SeqMixerOccCANINE,
    load_tokenizer
    )

from .dataloader import (
    OccDatasetV2FromAlreadyLoadedInputs
    )

from histocc.formatter import (
    blocky5,
    BOS_IDX,
)

from histocc.utils import Averager
from histocc.utils.decoder import (
    flat_decode_flat_model,
    flat_decode_mixer,
    greedy_decode, 
    mixer_greedy_decode,
    )

from .dataloader import concat_string_canine, OCCDataset, labels_to_bin, train_test_val, save_tmp, create_data_loader
from .trainer import trainer_loop_simple, eval_model
from .attacker import AttackerClass


def load_keys() -> pd.DataFrame:
    fn_keys = files('histocc').joinpath('Data/Key.csv')

    with fn_keys.open() as file:
        keys = pd.read_csv(file, skiprows=[1])

    return keys

# Get_adapted_tokenizer
def get_adapated_tokenizer(name: str):
    """
    This function loads the adapted tokenizers used in training
    """
    if "CANINE" in name:
        tokenizer = load_tokenizer("Multilingual_CANINE")
    else:
        tokenizer_save_path = name + '_tokenizer'
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_save_path)

    return tokenizer


def top_n_to_df(result, top_n: int) -> pd.DataFrame:
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

    # Return as DataFrame
    return pd.DataFrame(rows, columns=column_names)


class OccCANINE:
    def __init__(self, name = "OccCANINE", device = None, batch_size = 256, verbose = False, baseline = False, hf = True, force_download = False):
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
        self.tokenizer = get_adapated_tokenizer("CANINE_Multilingual_CANINE_sample_size_10_lr_2e-05_batch_size_256")

        # Get key
        self.key, self.key_desc = self._load_keys()
        
        # Formatter
        self.formatter = blocky5() # TODO: Handle other model specs
        
        # Model and model type
        self.model, self.model_type = self._load_model(hf, force_download, baseline)        
        
        # Prediction type
        self.prediction_type = None # Will be changed in any prediction
        
        # Promise for later initialization
        self.finetune_key = None
        
        # Max seq len: Maybe don't make this an arg? Changig it to something longer would require retraining?
        self.max_seq_len = 128

    def _load_keys(self) -> Tuple[Dict[float, str], Dict[float, str]]:
        # Load and return both the key and key with descriptions
        key_df = DATASETS['keys']()

        key = dict(zip(key_df.code, key_df.hisco))
        key_desc = dict(zip(key_df.code, key_df.en_hisco_text))

        return key, key_desc

    def _load_model(self, hf, force_download, baseline):
        
        # TODO: Implement loading form hugging face. Inserting original code below:
        # if hf:
            # model = CANINEOccupationClassifier_hub.from_pretrained(f"christianvedel/{self.name}", force_download=force_download).to(self.device)
            # model.to(self.device)
        
        # Load state
        model_path = f'{self.name}'
        loaded_state = torch.load(model_path, map_location=self.device)
        
        # Determine model type
        model_type = self._derive_model_type(loaded_state)
        
        # Load depending on model type
        if model_type == 'flat':
            model = CANINEOccupationClassifier(
                model_domain="Multilingual_CANINE", 
                n_classes = len(self.key), dropout_rate=0
                )
                
        elif model_type == 'seq2seq':
            model = Seq2SeqOccCANINE(
                model_domain='Multilingual_CANINE', # TODO make arg, discuss with Vedel
                num_classes=self.formatter.num_classes,
            )
            
        elif model_type == 'mix':
            model = Seq2SeqMixerOccCANINE(
                model_domain='Multilingual_CANINE', # TODO make arg, discuss with Vedel
                num_classes=self.formatter.num_classes,
                num_classes_flat = len(self.key), # TODO make arg or infer from formatter
            )
        else:
            raise NotImplementedError("Somehow an undefined 'model_type' was used")
        
        # Load params to model if not baseline:
        if not baseline:
            if model_type == 'flat':
                model.load_state_dict(loaded_state)
            else: 
                model.load_state_dict(loaded_state['model'])
                        
        model.to(self.device)
        
        return model, model_type
        
    def _derive_model_type(self, loaded_state):
        """
        Derives the model type 'flat', 'seq2seq' or 'mix' based on model arch
        """
        
        error_message = "Model type could not be automatically identified"
        
        # Determine model type
        if len(loaded_state) > 100: # If very long it is probably the old
            # OLD CASE:
            loaded_state_keys = loaded_state.keys()
            if 'basemodel.pooler.dense.bias' in loaded_state_keys:
                model_type = 'flat'
            else:
                raise NotImplementedError(error_message)
        elif len(loaded_state) == 4:
            # NEW CASE
            _ = 1
            model_dict_keys = loaded_state['model'].keys()
            
            if 'linear_decoder.bias' in model_dict_keys:
                model_type = 'mix'
            elif 'decoder.head.weight' in model_dict_keys: 
                model_type = 'seq2seq'
            else:
                raise NotImplementedError(error_message)
        else:
            raise NotImplementedError(error_message)
        
        
        return model_type
            
    def predict(self, occ1, lang = "unk", what = "pred", threshold = 0.22, concat_in = False, get_dict = False, get_df = True, behavior = "good", prediction_type = None, k_pred = 5):
        """
        Makes predictions on a batch of occupational strings.

        Parameters:
        - occ1 (list or str): A list of occupational strings to predict.
        - lang (str, optional): The language of the occupational strings. Defaults to "unk" (unknown).
        - what (str or int, optional): Specifies what to return. Options are "logits", "probs", "pred", "bin". Defaults to "pred".
        - threshold (float, optional): The prediction threshold for binary classification tasks. Defaults to 0.22. Which is generally optimal for F1.
        - concat_in (bool, optional): Specifies if the input is already concatenated in the format [occupation][SEP][language]. Defaults to False.
        - get_dict (bool, optional): If True and 'what' is an integer, returns a list of dictionaries with predictions instead of a DataFrame. Defaults to False.
        - get_df (bool, optional): If True and 'what' equals "pred", returns predictions in a DataFrame format. Defaults to True.
        - behavior (str): Simple argument to set prediction arguments. Should prediction be "good" or "fast"? Defaults to "good".  See details.
        - prediction_type (str): Either 'flat', 'greedy', 'full'. Overwrites 'behavior'. See details.
        - k_pred (int): Maximum number of predicted occupational codes to keep
        
        **Details.**
        *behvaior* 
        When 'fast' is chosen, the prediction will be based on a simple 'flat' decoder with one output neuron per possible class.
        When 'good' is chosen, the prediction will be based on a seq2seq transformer decoder. 
        The 'good' option is in the order of 5-10 times slower than the 'fast' option but performance is worse. 
        Se the paper for more details https://arxiv.org/abs/2402.13604
        
        *prediciton_type*
        The output from the CANINE transformer model needs to be turned into predictions. This option allows you to pick how you want this to happen.
        'flat' is the simplest. This takes the pooled output and feeds it into a single layer of output neurons with one output for each HISCO code.
        'greedy' runs the seq2seq transformer decoder in a greedy fashion. I.e. picking the most likely digit at each step.
        'full' evaluates all possible digit combinations through the seq2seq decoder and returns a probability of each of all the possible HISCO codes.
        Some 'prediction_type' options are not available for certain model types. This method will throw an error in those cases. 

        Returns:
        - Depends on the 'what' parameter. Can be logits, probabilities, predictions, a binary matrix, or a DataFrame containing the predicted classes and probabilities.
        """
        # Validate prediction arguments' compatability
        prediction_type = self._validate_and_update_prediction_parameters(behavior, prediction_type)
        
        # Handle list vs str
        if isinstance(occ1, str):
            occ1 = [occ1]
        
        # Clean string
        occ1 = self._prep_str(occ1)
        
        # Data loader
        dataset = OccDatasetV2FromAlreadyLoadedInputs(
            inputs = occ1,
            lang = lang,
            fname_index=1, # Dummy argument
            formatter = self.formatter,
            tokenizer = self.tokenizer,
            max_input_len=128,
            training=False,
        )
        
        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False
            )
                
        # Timing
        start = time.time()
        
        # Run prediction type
        if prediction_type == 'flat':
            out, out_type, inputs = self._predict_flat(data_loader)
        if prediction_type == 'greedy': 
            out, out_type, inputs = self._predict_greedy(data_loader)
        if prediction_type == 'full':
            out, out_type, inputs = self._predict_full(data_loader)
            
        # Return format
        result = self._format(out, out_type, what, inputs, lang, threshold, k_pred)
        
        
        # Time keeping
        end = time.time()
        if self.verbose:
            self._end_message(start, end, occ1)
        
        # Return
        return result
        
    def _predict_flat(self, data_loader): # TODO: Make sure it also works for model_type="mix"
        """
        Makes predictions on a batch of occupational strings.

        Parameters:
        - data_loader (DataLoader)

        Returns:
        - Depends on the 'what' parameter. Can be logits, probabilities, predictions, a binary matrix, or a DataFrame containing the predicted classes and probabilities.
        """
        model = self.model.eval()
        
        # Setup
        verbose = self.verbose
        results = []
        inputs = []
        total_batches = len(data_loader)
        
        batch_time = Averager()
        batch_time_data = Averager()
        
        # Need to initialize first "end time", as this is
        # calculated at bottom of batch loop
        end = time.time()
        
        # Decoder based on model type
        if self.model_type == "mix":
            decoder = flat_decode_mixer
        elif self.model_type == "flat":
            decoder = flat_decode_flat_model
        else:
            raise TypeError(f"model_type: '{self.model_type}' does not work with the flat prediciton")

        for batch_idx, batch in enumerate(data_loader, start=1):

            input_str = batch['occ1']
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            
            batch_time_data.update(time.time() - end)

            with torch.no_grad():
                output = decoder(
                    model = model,
                    descr = input_ids,
                    input_attention_mask = attention_mask
                    )
             
            # Store input in its original string format
            inputs.extend(batch['occ1'])    
            
            batch_time.update(time.time() - end)
                
            if batch_idx % 1 == 0 and verbose:
                print(f'Finished prediction for batch {batch_idx} of {total_batches}')
                print(f'Batch time (data): {batch_time.avg:.2f} ({batch_time_data.avg:.2f}).')
                print(f'Max. memory allocated/reserved: {torch.cuda.max_memory_allocated() / (1024 ** 3):.2f}/{torch.cuda.max_memory_reserved() / (1024 ** 3):.2f} GB')
            
            end = time.time()
            
            batch_logits = output
            batch_predicted_probs = torch.sigmoid(batch_logits).cpu().numpy()
            results.append(batch_predicted_probs)
            
        
        out_type = 'probs'
        
        return results, out_type, inputs
        
    @torch.no_grad
    def _predict_greedy(self, data_loader):
        model = self.model.eval()

        inputs = []

        preds_s2s_raw = []
        probs_s2s_raw = []

        batch_time = Averager()
        batch_time_data = Averager()
                        
        # Need to initialize first "end time", as this is
        # calculated at bottom of batch loop
        end = time.time()
        
        # Decoder based on model type
        if self.model_type == "mix":
            decoder = mixer_greedy_decode
        elif self.model_type == "seq2seq":
            decoder = greedy_decode
        else:
            raise TypeError(f"model_type: '{self.model_type}' does not work with the greedy prediciton")
        
        # Setup
        verbose = self.verbose
        total_batches = len(data_loader)

        for batch_idx, batch in enumerate(data_loader, start=1):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            
            batch_time_data.update(time.time() - end)
            
            outputs = decoder(
                model = model,
                descr = input_ids,
                input_attention_mask = attention_mask,
                device = self.device,
                max_len = data_loader.dataset.formatter.max_seq_len,
                start_symbol = BOS_IDX,
                )
            outputs_s2s = outputs[0].cpu().numpy()
            probs_s2s = outputs[1].cpu().numpy()
            
            # Store input in its original string format
            inputs.extend(batch['occ1'])
            
            # Store predictions
            preds_s2s_raw.append(outputs_s2s)
            probs_s2s_raw.append(probs_s2s)

            batch_time.update(time.time() - end)
            
            if batch_idx % 1 == 0 and verbose:
                print(f'Finished prediction for batch {batch_idx} of {total_batches}')
                print(f'Batch time (data): {batch_time.avg:.2f} ({batch_time_data.avg:.2f}).')
                print(f'Max. memory allocated/reserved: {torch.cuda.max_memory_allocated() / (1024 ** 3):.2f}/{torch.cuda.max_memory_reserved() / (1024 ** 3):.2f} GB')
            
            end = time.time()

        preds_s2s_raw = np.concatenate(preds_s2s_raw)
        probs_s2s_raw = np.concatenate(probs_s2s_raw)
        
        preds_s2s = list(map(
            data_loader.dataset.formatter.clean_pred,
            preds_s2s_raw,
        ))

        preds = pd.DataFrame({
            'input': inputs,
            'pred_s2s': preds_s2s,
            **{f'prob_s2s_{i}': probs_s2s_raw[:, i] for i in range(probs_s2s_raw.shape[1])},
        })
        
        out_type = 'greedy'
        
        return preds, out_type, inputs
            
    def _validate_and_update_prediction_parameters(self, behavior, prediction_type):
        """
        Wraps all the validation and updating of 'behavior' and 'prediction_type'
        and makes sure that they are compatible with 'self.model_type'

        Parameters:
        - behavior (str): Chosen behavior (see docstring of '.predict()')
        - prediction_type (str or None): Chosen prediction type (see docstring of '.predict()')

        Returns:
        - Possibly updated 'prediction_type'
        """
        
        # Validate 'behavior'
        test = behavior in ['good', 'fast']
        if not test:
            raise NotImplementedError(f"behavior: '{behavior}' is not implemented")
            
        # Validate 'behavior'
        test = self._behavior_compatible(behavior)
        if not test:
            raise NotImplementedError(f"behavior: '{behavior}' is not implemented for the loaded model, which has model type: '{self.model_type}'. Please specify different model in initialization or change 'behavior'.")
            
        # Set 'prediction_type' based on 'behavior'
        # If 'prediction_type' is not None then that prediction type overwrites
        if prediction_type is not None:
            _ = 1 # Change nothing
        else:
            if behavior == "fast":
                prediction_type = "flat"
            if behavior == "good":
                prediction_type = "greedy"
            
            print(f"Based on chosen 'behavior' ({behavior}) 'prediction_type' ({prediction_type}) was automatically set to '{prediction_type}'")
        
        # Validate 'prediction_type'
        test = prediction_type in ['flat', 'greedy', 'full']
        if not test:
            raise NotImplementedError(f"prediction_type: '{prediction_type}' is not implemented")
            
        # Validate prediction type is compatible with model type
        test = self._prediction_type_compatible(prediction_type)
        if not test:
            raise NotImplementedError(f"There is not implemented solution for handling: prediction_type: '{prediction_type}' togehter with model_type: '{self.model_type}'")
        self.prediction_type = prediction_type
        
        return prediction_type
        
            
    def _prediction_type_compatible(self, prediction_type):
        """
        Makes sure that the chosen combination of prediction type and model type
        are compatible

        Parameters:
        - prediction_type (str): Prediction type: 'flat', 'greedy', 'full'
        
        """        
        if self.model_type == "mix":
            res = True # Then all prediction types are possible
        elif self.model_type == "seq2seq":
            if prediction_type in ['greedy', 'full']: # Valid types for seq2seq
                res = True
            else:
                res = False
        elif self.model_type == "flat":
            if prediction_type in ['flat']: # Valid types for flat
                res = True
            else:
                res = False
        else:
            raise NotImplementedError(
                """
                This should not be possible. You did something weird to end up here.
                An invalid 'prediction_type' was used but somehow passed the first 
                check. 
                """
                )
        
        return res
    
    def _behavior_compatible(self, behavior):
        """
        Makes sure that the chosen combination of prediction type and model type
        are compatible

        Parameters:
        - behavior (str): Behavior type: 'good' or 'fast'
        
        """
        if not behavior in ['good', 'fast']:
            raise NotImplementedError(
                """
                This should not be possible. You did something weird to end up here.
                An invalid 'behavior' was used but somehow passed the first 
                check. 
                """
                )
        
        if self.model_type == "mix":
            res = True # Then all prediction types are possible
        elif self.model_type == "seq2seq":
            if behavior in ['good']: # Valid types for seq2seq
                res = True
            else:
                res = False
        elif self.model_type == "flat":
            if behavior in ['fast']: # Valid types for flat
                res = True
            else:
                res = False
        else:
            raise NotImplementedError(
                """
                This should not be possible. You did something weird to end up here.
                An invalid 'behavior' was used but somehow passed the first 
                check. 
                """
                )
        
        return res
    
    def _prep_str(self, occ1):
        """
        Prepares occupational strings into a format suitable for model input.

        Parameters:
        - occ1 (list of str): A list of occupational strings to encode.

        Returns:
        - list of str: Strings which are cleaned form non standard characters and in lower case
        """

        occ1 = [occ.lower() for occ in occ1]
        occ1 = [unidecode(occ) for occ in occ1]

        return occ1
    
    @staticmethod
    def _end_message(start, end, inputs):
        dif_time = end - start
        m, s = divmod(dif_time, 60)
        h, m = divmod(m, 60)

        try:
            # Assuming occ1 is a DataFrame
            nobs = inputs.shape[0]
        except AttributeError:
            # Fallback if occ1 does not have a .shape attribute, use len() instead
            nobs = len(inputs)

        print(f"Produced HISCO codes for {nobs} observations in {h:.0f} hours, {m:.0f} minutes and {s:.3f} seconds.")

        saved_time = nobs*10 - dif_time
        m, s = divmod(saved_time, 60)
        h, m = divmod(m, 60)

        print("Estimated hours saved compared to human labeller (assuming 10 seconds per label):")
        print(f" ---> {h:.0f} hours, {m:.0f} minutes and {s:.0f} seconds")
    
    def _format(self, out, out_type, what, inputs, lang, threshold, k_pred):
        """ 
        Formats preditions based on out, out_type and 'what'

        Parameters:
        - out: Model output
        - out_type: What is the output type? "greedy" or "probs"
        - what (str or int, optional): Specifies what to return. Options are "probs", "pred", "bin", or an integer [n] to return top [n] predictions. Defaults to "pred".
        - threshold (float): threshold to use (only relevant if out_type == "probs")
        - k_pred (int): Maximum number of predicted occupational codes to keep

        Returns:
        - Depends on the 'what' parameter.
        """
        
        if out_type == "probs":
            
            if what == "probs":
                # Unnest
                res = np.vstack(out)
                
                # Validate shape
                assert res.shape[0] == len(inputs), "N rows in inputs should equal N rows in output"
                assert res.shape[1] == len(self.key), "N cols should equal number of entries in self.key"
                                
            elif what == "pred":
                res = []
                for probs in out:
                    for row in probs:
                        topk_indices = np.argsort(row)[-k_pred:][::-1]
                        row = [[self.key[i], row[i], self.key_desc[i]] for i in topk_indices]
                        row = [item for sublist in row for item in sublist] # Flatten list
                        res.append(row)
                                                
                column_names = []
                for i in range(1, k_pred+1):
                    column_names.extend([f'hisco_{i}', f'prob_{i}', f'desc_{i}'])
                               
                res = pd.DataFrame(res, columns=column_names)
                
                # Vectorized operation to mask predictions below the threshold
                for j in range(1, k_pred+1):
                    prob_column = f"prob_{j}"
                    mask = res[prob_column] <= threshold
                    res.loc[mask, [f"hisco_{j}", f"desc_{j}", f"prob_{j}"]] = [float("NaN"), "No pred", float("NaN")]
            
                # First, ensure "hisco_1" is of type string to avoid mixing data types
                res["hisco_1"] = res["hisco_1"].astype(str)
                res["hisco_1"].fillna("-1", inplace=True)
                
                res.insert(0, 'occ1', inputs)
            
            else:
                raise ValueError(f"'what' ('{what}') did not match any output for 'out_type' ('{out_type}')")
                
                
        elif out_type == "greedy":
            res = out # TODO: Turn into same format as above. 
       
        
       
        return res
               
    
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
            inputs = [concat_string_canine(occ, l) for occ, l in zip(occ1, lang)]

        return inputs
        
    def predict_old(self, occ1, lang = "unk", what = "pred", threshold = 0.22, concat_in = False, get_dict = False, get_df = True):
        """
        Makes predictions on a batch of occupational strings.

        Parameters:
        - occ1 (list or str): A list of occupational strings to predict.
        - lang (str, optional): The language of the occupational strings. Defaults to "unk" (unknown).
        - what (str or int, optional): Specifies what to return. Options are "logits", "probs", "pred", "bin", or an integer [n] to return top [n] predictions. Defaults to "pred".
        - threshold (float, optional): The prediction threshold for binary classification tasks. Defaults to 0.22. Which is generally optimal for F1.
        - concat_in (bool, optional): Specifies if the input is already concatenated in the format [occupation][SEP][language]. Defaults to False.
        - get_dict (bool, optional): If True and 'what' is an integer, returns a list of dictionaries with predictions instead of a DataFrame. Defaults to False.
        - get_df (bool, optional): If True and 'what' equals "pred", returns predictions in a DataFrame format. Defaults to True.

        Returns:
        - Depends on the 'what' parameter. Can be logits, probabilities, predictions, a binary matrix, or a DataFrame containing the predicted classes and probabilities.
        """
        
        # Handle list vs str
        if isinstance(occ1, str):
            occ1 = [occ1]
        
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

        if get_df and what=="pred":
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
                raise ValueError("'what' incorrectly specified")

        # Clean results
        if what == "bin":
            results = np.array(results)

        if what == "probs":
            results = np.concatenate(results, axis=0)

        if isinstance(what, (int, float)):
            if not get_dict:
                results = top_n_to_df(results, what)
            
        if isinstance(what, (int, float)) and what0 == "pred":
            print("\nPrediction done. Cleaning results.")
        
            # Vectorized operation to mask predictions below the threshold
            for j in range(1, what + 1):
                prob_column = f"prob_{j}"
                mask = results[prob_column] <= threshold
                results.loc[mask, [f"hisco_{j}", f"desc_{j}", f"prob_{j}"]] = [float("NaN"), "No pred", float("NaN")]
        
            # First, ensure "hisco_1" is of type string to avoid mixing data types
            results["hisco_1"] = results["hisco_1"].astype(str)
            results["hisco_1"].fillna("-1", inplace=True)
            
            results.insert(0, 'inputs', inputs)

        end = time.time()

        if verbose:
            dif_time = end - start
            m, s = divmod(dif_time, 60)
            h, m = divmod(m, 60)

            try:
                # Assuming occ1 is a DataFrame
                nobs = occ1.shape[0]
            except AttributeError:
                # Fallback if occ1 does not have a .shape attribute, use len() instead
                nobs = len(occ1)

            print(f"Produced HISCO codes for {nobs} observations in {h:.0f} hours, {m:.0f} minutes and {s:.3f} seconds.")

            saved_time = nobs*10 - dif_time
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
            # breakpoint()
            key = self.key

            # # Remove na items
            # items = list(key.items())
            # items.pop(0)
            # key = dict(items)

            # Convert the key dictionary to a DataFrame for joining
            key_df = pd.DataFrame(list(key.items()), columns=['Code', 'Hisco'])
            # Convert 'Hisco' in key_df to numeric (float), as it's going to be merged with numeric columns
            key_df['Hisco'] = pd.to_numeric(key_df['Hisco'], errors='coerce')
            # Ensure the label columns are numeric and have the same type as the key
            for col in label_cols:
                data_df[col] = pd.to_numeric(data_df[col], errors='coerce')

            self.finetune_key = key

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
        df_train, df_val = train_test_val(data_df, verbose=False, testval_fraction = testval_fraction, test_size = 0)

        # To use later
        n_obs_train = df_train.shape[0]
        n_obs_val = df_val.shape[0]

        # Print split sizes
        print(f"{n_obs_train} observations will be used in training.")
        print(f"{n_obs_val} observations will be used in validation.")

        # Save tmp files
        save_tmp(df_train, df_val, df_test = df_val, path = "Data/Tmp_finetune") # FIXME avoid hardcoded paths

        # File paths for the index files
        train_index_path = "Data/Tmp_finetune/Train_index.txt" # FIXME avoid hardcoded paths
        val_index_path = "Data/Tmp_finetune/Val_index.txt" # FIXME avoid hardcoded paths

        # Calculate number of classes
        n_classes = len(key)

        # Instantiating OCCDataset with index file paths
        ds_train = OCCDataset(df_path="Data/Tmp_finetune/Train.csv", n_obs=n_obs_train, tokenizer=tokenizer, attacker=attacker, max_len=128, n_classes=n_classes, index_file_path=train_index_path, alt_prob=0, insert_words=False, model_domain=model_domain, unk_lang_prob = 0) # FIXME avoid hardcoded paths
        ds_train_attack = OCCDataset(df_path="Data/Tmp_finetune/Train.csv", n_obs=n_obs_train, tokenizer=tokenizer, attacker=attacker, max_len=128, n_classes=n_classes, index_file_path=train_index_path, alt_prob=alt_prob, insert_words=insert_words, model_domain=model_domain, unk_lang_prob = 0) # FIXME avoid hardcoded paths
        ds_val = OCCDataset(df_path="Data/Tmp_finetune/Val.csv", n_obs=n_obs_val, tokenizer=tokenizer, attacker=attacker, max_len=128, n_classes=n_classes, index_file_path=val_index_path, alt_prob=0, insert_words=False, model_domain=model_domain, unk_lang_prob = 0) # FIXME avoid hardcoded paths

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


    def _train_model(self, processed_data, model_name, epochs, only_train_final_layer, verbose = True, verbose_extra = False, new_labels = False, save_model = True, save_path = '../OccCANINE/Finetuned/'):
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
        save_model : bool, optional
            Should the finetuned model be saved? Defaults to false

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
            initial_loss = val_loss,
            save_model = save_model,
            save_path = save_path
            )

        # == Load best model ==
        # breakpoint()
        # Load state
        if save_model:
            model_path = save_path + model_name+'.bin'
            if not os.path.isfile(model_path):
                print("Model did not improve in training. Realoding original model")
                model_path = self.name+'.bin'

            # Load the model state
            loaded_state = torch.load(model_path)

            # If-lookup for model
            config = {
                "model_domain": "Multilingual_CANINE",
                "n_classes": len(self.finetune_key),
                "dropout_rate": 0,
                "model_type": "canine"
            }

            model = CANINEOccupationClassifier_hub(config)
            # breakpoint()
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
            new_labels = False,
            save_model = True,
            save_path = "Finetuned/"
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
        save_model : bool, optional
            Whether the finetuned model should be saved.
        save_path : str, optional
            Where should the model be saved?

        Returns
        -------
        None
        """

        print("======================================")
        print("==== Started finetuning procedure ====")
        print("======================================")

        # Make sure 'Finetuned' dir exist
        if not os.path.exists(save_path):
            # Create the directory
            os.makedirs(save_path)

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
            new_labels = new_labels,
            save_model = save_model,
            save_path = save_path
            )

        print("Finetuning completed successfully.")
