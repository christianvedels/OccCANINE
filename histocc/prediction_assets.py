# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 23:03:44 2023
Loads trained version of the models
@author: chris
"""


import os
import time
import warnings

from typing import Dict, Tuple, Literal

import torch

from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from unidecode import unidecode

import numpy as np
import pandas as pd

from .loss import (
    BlockOrderInvariantLoss,
    LossMixer,
)
from .datasets import DATASETS
from .model_assets import (
    CANINEOccupationClassifier,
    CANINEOccupationClassifier_hub,
    Seq2SeqOccCANINE,
    Seq2SeqMixerOccCANINE,
    Seq2SeqMixerOccCANINE_hub,
    Seq2SeqOccCANINE_hub,
    load_tokenizer
    )
from .dataloader import (
    OccDatasetV2FromAlreadyLoadedInputs,
    )
from .formatter import (
    hisco_blocky5,
    BOS_IDX,
    PAD_IDX,
    construct_general_purpose_formatter,
)
from .utils import (
    Averager,
    load_states,
    prepare_finetuning_data,
    setup_finetuning_datasets,
)
from .utils.decoder import (
    flat_decode_flat_model,
    flat_decode_mixer,
    greedy_decode,
    mixer_greedy_decode,
    full_search_decoder_seq2seq_optimized,
    full_search_decoder_mixer_optimized,
    )
from .seq2seq_mixer_engine import train
from itertools import permutations


PredType = Literal['flat', 'greedy', 'full']
SystemType = Literal['hisco'] | str
BehaviorType = Literal['good', 'fast']
ModelType = Literal['flat', 'seq2seq', 'mix']
ModelName = Literal['OccCANINE', 'OccCANINE_s2s', 'OccCANINE_s2s_mix']

def load_keys() -> pd.DataFrame:
    ''' Load dictionary mapping between HISCO codes and {0, 1, ..., k} format
    '''
    return DATASETS['keys']()


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
        column_names.extend([f'{self.system}_{i}', f'prob_{i}', f'desc_{i}'])

    # Return as DataFrame
    return pd.DataFrame(rows, columns=column_names)


class OccCANINE:
    def __init__(
            self,
            name: ModelName | str = "OccCANINE_s2s_mix",
            device: torch.device | None = None,
            batch_size: int = 256,
            verbose: bool = True,
            baseline: bool = False,
            hf: bool = True,
            force_download: bool = False,
            system: SystemType = "hisco",
            # args primarily for testing purposes -- want to instantiate without loading
            model_type: ModelType | None = None,
            skip_load: bool = False,
            # Args used for other systems
            descriptions: pd.DataFrame | None = None,
            use_within_block_sep: bool = False, # Should be True for systems with ',' between digits
            target_cols: list[str] | None = None,
    ):
        """
        Initializes the OccCANINE model with specified configurations.

        Parameters:
        - name (str): Name of the model to load. Defaults to "OccCANINE_s2s_mix". For local models, specify the model name as it appears and set 'hf' to False.
        - device (str or None): The computing device (CPU or CUDA) on which the model will run. If None, the device is auto-detected.
        - batch_size (int): The number of examples to process in each batch. This affects memory usage and processing speed.
        - verbose (bool): If True, progress updates and diagnostic information will be printed during model operations.
        - baseline (bool): If True, loads a baseline (untrained) version of CANINE. Useful for comparison studies.
        - hf (bool): If True, attempts to load the model from Hugging Face's model repository. If False, loads a local model specified by the 'name' parameter.
        - force_download (bool): If True, forces a re-download of the model from the Hugging Face's model repository even if it is already cached locally.
        - system (str): Which encoding system is it? For now this only works for "HISCO"
        - descriptions (pd.DataFrame): A DataFrame with two columns: 1) codes in some system 'system_code', and 2) their corresponding descriptions, 'desc'. Only used for none-HISCO predictions.

        Raises:
        - Exception: If 'hf' is False and a local model 'name' is not provided.
        """

        # Check if model name is provided when not using Hugging Face
        if name in ModelName.__args__ and not hf and not skip_load:
            raise ValueError("When 'hf' is False, a specific local model 'name' must be provided.")


        # Detect device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)

        if verbose:
            print(f"Using device: {self.device}")

        self.name = name
        self.batch_size = batch_size
        self.verbose = verbose

        # Get tokenizer
        self.tokenizer = get_adapated_tokenizer("CANINE_Multilingual_CANINE_sample_size_10_lr_2e-05_batch_size_256") # Universal tokenizer

        # System
        self.system = system
        self.use_within_block_sep = use_within_block_sep

        if self.system == "hisco": # TODO: Handle other model specs
            # Formatter
            self.formatter = hisco_blocky5()

            # Get key
            self.key, self.key_desc = self._load_keys()

            # Length of codes
            self.code_len = 5

            # List of codes formatted to fit with the output from seq2seq/mix model
            self.codes_list = self._list_of_formatted_codes()
        else:
            if hf:
                raise ValueError("Hugging Face loading is only supported for the 'HISCO' system. Please set 'hf' to False and provide a local model name.")

            # TODO: Move into key loading method
            loaded_state = torch.load(name, weights_only=True) # Load state
            key_loaded = loaded_state['key']
            self.key = {int(v): str(k) for k, v in key_loaded.items()} # Invert key and cast to int / str
            self.key_desc = {k: "Not provided" for k in self.key.keys()}

            if descriptions is not None:
                # Test of correct colummns are in descriptions
                if not all([i in descriptions.columns for i in ['system_code', 'desc']]):
                    raise ValueError("The descriptions must contain the columns 'system_code' and 'desc'")

                # Desc dict
                desc_dict = descriptions.set_index('system_code')['desc'].to_dict()
                desc_dict = {str(k): str(v) for k, v in desc_dict.items()}

                # Join on descriptions
                self.key_desc = {k: desc_dict.get(self.key.get(k), "Not provided") for k in self.key.keys()} # Produce key_desc

            # Code len as max of keys
            key_val_tmp = list(self.key.values())
            if self.use_within_block_sep:
                self.code_len = max([len(i.split(",")) for i in key_val_tmp])
            else:
                self.code_len = max([len(i) for i in self.key.values()])

            # Load general purpose formatter
            target_cols = target_cols if target_cols is not None else ["dummy_entry"] * 5
            if "chars" in loaded_state.keys():
                self.formatter = construct_general_purpose_formatter(block_size=self.code_len, target_cols=target_cols, chars=loaded_state['chars'])
            else:
                self.formatter = construct_general_purpose_formatter(block_size=self.code_len, target_cols=target_cols, use_within_block_sep=use_within_block_sep)

            # Check if use_within_block_sep is set
            if not use_within_block_sep:
                if any(["," in i for i in self.key.values()]):
                    raise ValueError("The key contains ',' in some of the codes. Please set 'use_within_block_sep' to True")

            # List of codes formatted to fit with the output from seq2seq/mix model
            self.codes_list = self._list_of_formatted_codes()

        # Sanitize keys (system codes should be of self.code_len + int as keys)
        if use_within_block_sep:
            self.key = {int(k): str(v) for k, v in self.key.items()}
        else:
            self.key = {int(k): str(v).zfill(self.code_len) for k, v in self.key.items()}

        self.key_desc = {int(k): str(v) for k, v in self.key_desc.items()}

        # Model and model type
        if skip_load:
            if model_type is None:
                raise ValueError('Must specify `model_type` if not loading weights')

            self.model_type = model_type
            self.model = self._initialize_model(model_type=model_type)
        else:
            if model_type is not None:
                warnings.warn('specified model_type, but discarding argument as model leading specified; model_type will be inferred')

            self.model, self.model_type = self._load_model(hf, force_download, baseline)

        # Prediction type
        self.prediction_type = None # Will be changed in any prediction

        # Promise for later initialization
        self.finetune_key = None

        # Max seq len: Maybe don't make this an arg? Changig it to something longer would require retraining?
        self.max_seq_len = 128

    def __repr__(self):
        return (
            f"OccCANINE("
            f"name='{self.name}', "
            f"device='{self.device}', "
            f"batch_size={self.batch_size}, "
            f"verbose={self.verbose}, "
            f"system='{self.system}', "
            f"model_type='{self.model_type}', "
            f"formatter='{self.formatter}')"
        )

    def __call__(self, occ1: str | list[str], *args, **kwargs):
        return self.predict(occ1, *args, **kwargs)

    def _load_keys(self, path: str | None = None) -> Tuple[Dict[float, str], Dict[float, str]]:

        if path is not None:
            key_df = pd.read_csv(path)

            # Test if the df contains the right columns
            if not all([i in key_df.columns for i in ['code', 'system_code', 'desc']]):
                raise ValueError("The key file does not contain the right columns. It must contain 'code', 'system_code' and 'desc'")

            # Produce the key and key_desc
            key = dict(zip(key_df.code, key_df.system_code))
            key_desc = dict(zip(key_df.code, key_df.desc))

        else:
            # Load and return both the key and key with descriptions
            key_df = DATASETS['keys']()

            key = dict(zip(key_df.code, key_df.hisco))
            key_desc = dict(zip(key_df.code, key_df.en_hisco_text))

        key = {str(k): str(v) for k, v in key.items()} # Invert key and cast to int / str
        key_desc = {str(k): str(v) for k, v in key_desc.items()} # Invert key and cast to int / str

        return key, key_desc

    def _list_of_formatted_codes(self, codes_list: list[str] | None = None):
        """
        Returns a list of formatted codes. According to the seq2seq formatter
        """
        # Get codes
        if codes_list is None:
            codes = self.key.values()
            codes_list = list(codes)

        # Formatted list of codes
        codes_list = [str(i) for i in codes_list]
        if not self.use_within_block_sep: # This cleaning step inserts erroneous 0 in the codes if use_within_block_sep is True
            codes_list = [i.zfill(self.code_len) if len(i) == (self.code_len-1) else i for i in codes_list]
        codes_list = [i for i in codes_list if i != " "]
        codes_list = [self.formatter.transform_label(i)[1:(1+self.code_len)] for i in codes_list]

        return codes_list

    def _initialize_model(
            self,
            model_type: ModelType,
            state_dict = None,
            baseline = False
            ) -> nn.Module:
        if model_type == 'flat':
            model = CANINEOccupationClassifier(
                model_domain="Multilingual_CANINE",
                n_classes = len(self.key), dropout_rate=0
                )

        elif model_type == 'seq2seq':
            model = Seq2SeqOccCANINE(
                model_domain='Multilingual_CANINE',
                num_classes=self.formatter.num_classes,
            )

        elif model_type == 'mix':
            model = Seq2SeqMixerOccCANINE(
                model_domain='Multilingual_CANINE',
                num_classes=self.formatter.num_classes,
                num_classes_flat = len(self.key),
            )
        else:
            raise ValueError(f"Undefined 'model_type' {model_type} requested")

        # Load params to model if not baseline:
        if baseline: # Strange but ensures backwards compatibility
            state_dict = None
        if state_dict is not None:
            if model_type == 'flat':
                model.load_state_dict(state_dict)
            else:
                model.load_state_dict(state_dict['model'])

        model.to(self.device)

        return model

    def _load_model(self, hf, force_download, baseline):
        # Load model from Hugging Face (only works for the old model))
        if hf:
            # Validate model name
            if self.name not in ["OccCANINE", "OccCANINE_s2s_mix", "OccCANINE_s2s"]:
                raise ValueError("Hugging Face loading is only supported for the 'OccCANINE', 'OccCANINE_s2s' and 'OccCANINE_s2s_mix' models.")

            # Load model based on name
            if self.name == "OccCANINE":
                model = CANINEOccupationClassifier_hub.from_pretrained(f"Christianvedel/{self.name}", force_download=force_download).to(self.device)
                model.to(self.device)
                model_type = "flat"

            elif self.name == "OccCANINE_s2s_mix":
                model = Seq2SeqMixerOccCANINE_hub.from_pretrained(f"Christianvedel/{self.name}", force_download=force_download).to(self.device)
                model.to(self.device)
                model_type = "mix"
            elif self.name == "OccCANINE_s2s":
                model = Seq2SeqOccCANINE_hub.from_pretrained(f"Christianvedel/{self.name}", force_download=force_download).to(self.device)
                model.to(self.device)
                model_type = "seq2seq"
            else:
                raise ValueError("Hugging Face loading is only supported for the 'OccCANINE', 'OccCANINE_s2s' and 'OccCANINE_s2s_mix' models.")


            return model, model_type

        # Load state
        model_path = f'{self.name}'
        loaded_state = torch.load(model_path, weights_only = True, map_location=self.device)

        # Determine model type
        model_type = self._derive_model_type(loaded_state)

        # Load depending on model type
        model = self._initialize_model(model_type=model_type, state_dict=loaded_state, baseline=baseline)

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
        elif len(loaded_state) < 10:
            # NEW CASE
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

    def predict(
            self,
            occ1: str | list[str],
            lang: str = "unk",
            what: str = "pred",
            threshold: float = 0.22,
            concat_in: bool = False,
            get_dict: bool = False,
            get_df: bool = True,
            behavior: BehaviorType = "good",
            prediction_type: PredType | None = None,
            k_pred: int = 5,
            order_invariant_conf: bool = True,
    ):
        """
        Makes predictions on a batch of occupational strings.

        Parameters:
        - occ1 (list or str): A list of occupational strings to predict.
        - lang (str, optional): The language of the occupational strings. Defaults to "unk" (unknown).
        - what (str or int, optional): Specifies what to return. Options are "logits", "probs", "pred", "bin". Defaults to "pred".
        - threshold (float, optional): The prediction threshold for binary classification tasks. Defaults to 0.22. Which is generally optimal for F1.
        - concat_in (bool, optional): Ignored
        - get_dict (bool, optional): Ignored
        - get_df (bool, optional): Ignored
        - behavior (str): Simple argument to set prediction arguments. Should prediction be "good" or "fast"? Defaults to "good".  See details.
        - prediction_type (str): Either 'flat', 'greedy', 'full'. Overwrites 'behavior'. See details.
        - k_pred (int): Maximum number of predicted occupational codes to keep
        - order_invariant_conf (bool): If True an order invariant confidence is computed. This takes a bit longer but - especially for cases with many observations with multiple occupations.

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

        More about the 'full' prediction type in self._predict_full

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
        elif prediction_type == 'greedy':
            out, out_type, inputs = self._predict_greedy(data_loader, order_invariant_conf=order_invariant_conf)
        elif prediction_type == 'full':
            out, out_type, inputs = self._predict_full(data_loader)
        else:
            raise ValueError(f'Unsupported prediction type {prediction_type}, must be one of {PredType}')

        # Return format
        result = self._format(out, out_type, what, inputs, lang, threshold, k_pred, order_invariant_conf)

        # Time keeping
        end = time.time()

        if self.verbose:
            self._end_message(start, end, occ1)

        # Return
        return result

    def _predict_flat(self, data_loader):
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
                print(f'\rFinished prediction for batch {batch_idx} of {total_batches}', end = "")

            end = time.time()

            batch_logits = output
            batch_predicted_probs = torch.sigmoid(batch_logits).cpu().numpy()
            results.append(batch_predicted_probs)

        results = np.concatenate(results)

        out_type = 'probs'

        return results, out_type, inputs

    @torch.no_grad()
    def _predict_greedy(self, data_loader, order_invariant_conf):
        model = self.model.eval()

        inputs = []

        preds_s2s_raw = []
        probs_s2s_raw = []
        if order_invariant_conf:
            order_inv_probs = []
        else:
            order_inv_probs = 1 # Placeholder

        batch_time = Averager()
        batch_time_data = Averager()

        # Need to initialize first "end time", as this is
        # calculated at bottom of batch loop
        end = time.time()

        # Decoder based on model type
        if self.model_type == "mix":
            decoder = mixer_greedy_decode
            decoder_full = full_search_decoder_mixer_optimized # Used in order invariant confidence
        elif self.model_type == "seq2seq":
            decoder = greedy_decode
            decoder_full = full_search_decoder_seq2seq_optimized # Used in order invariant confidence
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
            
            # Compute order invariant confidence
            if order_invariant_conf:
                # Location of multiple labels
                outputs_mapped_to_label = list(map(
                    data_loader.dataset.formatter.clean_pred,
                    outputs[0].cpu().numpy(),
                ))
                
                # Generate list of codes 
                codes_lists = [self._output_permutations(i) for i in outputs_mapped_to_label]

                order_inv_probs_batch = [float(0) for i in range(len(outputs_s2s))]

                for i, codes_list in enumerate(codes_lists):

                    len_list = len(codes_list)
                    if len_list > 1:
                        # Transform to machine readable representation:
                        codes_list_encoded = self._list_of_formatted_codes(codes_list = codes_list)

                        # Prepare subset tensors for the i-th sample
                        input_ids_i = input_ids[i].unsqueeze(0)
                        attention_mask_i = attention_mask[i].unsqueeze(0)

                        # Run the full search decoder on the current sample
                        outputs_order_inv = decoder_full(
                            model = model,
                            descr = input_ids_i,
                            input_attention_mask = attention_mask_i,
                            device = self.device,
                            codes_list = codes_list_encoded,
                            start_symbol = BOS_IDX,
                        )
                        # Test 
                        if outputs_order_inv.shape[1] != len_list:
                            raise ValueError(f"outputs_order_inv.shape[1] != len_list: {outputs_order_inv.shape[1]} != {len_list}")

                        # Take sum of the probabilities
                        order_inv_probs_batch[i] = float(outputs_order_inv.sum(axis=1).cpu().numpy()[0])

            # Store input in its original string format
            inputs.extend(batch['occ1'])

            # Store predictions
            preds_s2s_raw.append(outputs_s2s)
            probs_s2s_raw.append(probs_s2s)
            if order_invariant_conf:
                order_inv_probs.append(order_inv_probs_batch)

            batch_time.update(time.time() - end)

            if batch_idx % 1 == 0 and verbose:
                print(f'\rFinished prediction for batch {batch_idx} of {total_batches}', end = "")

            end = time.time()

        preds_s2s_raw = np.concatenate(preds_s2s_raw)
        probs_s2s_raw = np.concatenate(probs_s2s_raw)
        if order_invariant_conf:
            order_inv_probs = np.concatenate(order_inv_probs)

        preds_s2s = list(map(
            data_loader.dataset.formatter.clean_pred,
            preds_s2s_raw,
        ))

        if order_invariant_conf:
        
            preds = pd.DataFrame({
                'input': inputs,
                'pred_s2s': preds_s2s,
                **{f'prob_s2s_{i}': probs_s2s_raw[:, i] for i in range(probs_s2s_raw.shape[1])},
                'order_inv_conf': order_inv_probs,
            })
        else:
            preds = pd.DataFrame({
                'input': inputs,
                'pred_s2s': preds_s2s,
                **{f'prob_s2s_{i}': probs_s2s_raw[:, i] for i in range(probs_s2s_raw.shape[1])},
            })

        out_type = 'greedy'

        return preds, out_type, inputs

    @torch.no_grad()
    def _predict_full(self, data_loader):
        """
        This is the full prediction type. This takes all the codes in self.key and runs it through a seq2seq
        decoder. As such this in the order of 330 times slower than the typical the greedy decoder. But with
        the benefit that a probability of each code is returned.

        This rather larger increase in eval time is because the method requires, that we run all of the 1910
        HISCO codes through something akin to the greedy decoder. We achieve some speedup by only running the
        decoder on 5 digits.
        """
        model = self.model.eval()

        inputs = []

        batch_time = Averager()
        batch_time_data = Averager()

        # Need to initialize first "end time", as this is
        # calculated at bottom of batch loop
        end = time.time()

        # Decoder based on model type
        if self.model_type == "mix":
            decoder = full_search_decoder_mixer_optimized
        elif self.model_type == "seq2seq":
            decoder = full_search_decoder_seq2seq_optimized
        else:
            raise TypeError(f"model_type: '{self.model_type}' does not work with the greedy prediciton")

        # Setup
        verbose = self.verbose
        total_batches = len(data_loader)
        results = []

        for batch_idx, batch in enumerate(data_loader, start=1):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            batch_time_data.update(time.time() - end)

            output = decoder(
                model = model,
                descr = input_ids,
                input_attention_mask = attention_mask,
                device = self.device,
                codes_list = self.codes_list,
                start_symbol = BOS_IDX,
                )


            # Store input in its original string format
            inputs.extend(batch['occ1'])

            # Store predictions
            output_np = output.cpu().numpy()
            results.append(output_np)


            batch_time.update(time.time() - end)

            if batch_idx % 1 == 0 and verbose:
                print(f'\rFinished prediction for batch {batch_idx} of {total_batches}', end = "")

            end = time.time()

        results = np.concatenate(results)

        out_type = 'probs'

        return results, out_type, inputs
    
    def _output_permutations(self, output):
        """
        This function takes an output from the model with multiple labels
        and returns the permutations of the labels.
        This is used to compute the order invariant confidence.
        """

        output = output.split("&")
        if(len(output) == 1):
            return [output]

        # Generate all permutations of the output list
        permutations_list = list(permutations(output))

        # Join each permutation with the '&' symbol
        return ["&".join(permutation) for permutation in permutations_list]



    def _validate_and_update_prediction_parameters(self, behavior, prediction_type: PredType | None):
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
            pass # Change nothing
        else:
            if behavior == "fast":
                prediction_type = "flat"
            if behavior == "good":
                prediction_type = "greedy"

            if self.verbose: 
                print(f"Based on behavior = '{behavior}', prediction_type was automatically set to '{prediction_type}'")

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

    def _prediction_type_compatible(self, prediction_type: PredType):
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

        occ1 = [str(occ) for occ in occ1]
        occ1 = [occ.lower() for occ in occ1]
        occ1 = [unidecode(occ) for occ in occ1]

        return occ1

    def _end_message(self, start, end, inputs):
        dif_time = end - start
        m, s = divmod(dif_time, 60)
        h, m = divmod(m, 60)

        try:
            # Assuming occ1 is a DataFrame
            nobs = inputs.shape[0]
        except AttributeError:
            # Fallback if occ1 does not have a .shape attribute, use len() instead
            nobs = len(inputs)

        print(f"\nProduced {self.system} codes for {nobs} observations in {h:.0f} hours, {m:.0f} minutes and {s:.3f} seconds.")

        saved_time = nobs * 10 - dif_time
        m, s = divmod(saved_time, 60)
        h, m = divmod(m, 60)

        print("Estimated hours saved compared to human labeller (assuming 10 seconds per label):")
        print(f" ---> {h:.0f} hours, {m:.0f} minutes and {s:.0f} seconds")

        print("")
        print("If the time saved is valuable for you, please cite our paper:")
        self.citation()

    def citation(self, BibTex = False):
        if BibTex:
            print("@article{dahl2024breakinghisco,")
            print("  title={Breaking the HISCO Barrier: Automatic Occupational Standardization with OccCANINE},")
            print("  author={Christian Møller Dahl and Torben Johansen and Christian Vedel},")
            print("  year={2024}")
            print("  eprint={2402.13604},")
            print("  archivePrefix={arXiv},")
            print("  primaryClass={cs.CL},")
            print("  url={https://arxiv.org/abs/2402.13604},")
            print("}")
        else:
            print("Dahl, C. M., Johansen, T., & Vedel, C. (2024). Breaking the HISCO Barrier: Automatic Occupational Standardization with OccCANINE. arXiv preprint arXiv:2402.13604.")
            print("URL: https://arxiv.org/abs/2402.13604")
            print("")
            print("(You can get a BibTex citation by calling 'mod.citation(BibTex = True)')")

    def _format(
            self,
            out,
            out_type: str,
            what: str,
            inputs,
            lang: str,
            threshold: float,
            k_pred: int,
            order_invariant_conf: bool = True,
    ):
        """
        Formats preditions based on out, out_type and 'what'

        Parameters:
        - out: Model output
        - out_type: What is the output type? "greedy" or "probs"
        - what (str or int, optional): Specifies what to return. Options are "probs", "pred", "bin", or an integer [n] to return top [n] predictions. Defaults to "pred".
        - inputs (list): The original occupational strings used for prediction.
        - threshold (float): threshold to use (only relevant if out_type == "probs")
        - k_pred (int): Maximum number of predicted occupational codes to keep
        - order_invariant_conf (bool): If True an order invariant confidence is computed. 

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
                for row in out:
                    topk_indices = np.argsort(row)[-k_pred:][::-1]
                    row = [[self.key[i], row[i], self.key_desc[i]] for i in topk_indices]
                    row = [item for sublist in row for item in sublist] # Flatten list
                    res.append(row)

                column_names = []
                for i in range(1, k_pred+1):
                    column_names.extend([f'{self.system}_{i}', f'prob_{i}', f'desc_{i}'])

                res = pd.DataFrame(res, columns=column_names)

                # Vectorized operation to mask predictions below the threshold
                for j in range(1, k_pred + 1):
                    prob_column = f"prob_{j}"
                    mask = res[prob_column] <= threshold
                    res.loc[mask, [f"{self.system}_{j}", f"desc_{j}", f"prob_{j}"]] = [float("NaN"), "No pred", float("NaN")]

                # First, ensure "hisco_1" is of type string to avoid mixing data types
                res[f"{self.system}_1"] = res[f"{self.system}_1"].astype(str)

                res.insert(0, 'occ1', inputs)

            elif what == "bin":
                # Unnest
                res = np.vstack(out)

                # Validate shape
                assert res.shape[0] == len(inputs), "N rows in inputs should equal N rows in output"
                assert res.shape[1] == len(self.key), "N cols should equal number of entries in self.key"
                # Create matrix of zeros
                res = np.zeros((len(inputs), len(self.key)))
                for i, row in enumerate(out):
                    topk_indices = np.argsort(row)[-k_pred:][::-1]
                    for j in topk_indices:
                        if row[j] >= threshold:
                            res[i, j] = 1
                res = pd.DataFrame(res, columns=[f"{self.system}_{self.key[i]}" for i in range(len(self.key))])

            else:
                raise ValueError(f"'what' ('{what}') did not match any output for 'out_type' ('{out_type}')")

        elif out_type == "greedy":
            if what == "probs":
                raise ValueError("Probs cannot be computed for 'greedy' prediction_type. Use 'full' prediction_type instead")

            elif what in ["pred", "bin"]:
                sepperate_preds = [self._split_str_s2s(i) for i in out.pred_s2s]
                max_elements = max(len(item) if isinstance(item, list) else 1 for item in sepperate_preds)

                # Create an empty list to store the processed data
                processed_data = []

                # Process the data
                for item in sepperate_preds:
                    if isinstance(item, list):
                        # If the item is a list, unpack its elements and pad with NaN if necessary
                        processed_data.append(item + [np.nan] * (max_elements - len(item)))
                    else:
                        # If the item is not a list, append it with NaN for the remaining columns
                        processed_data.append([item] + [np.nan] * (max_elements - 1))

                # Invert key
                inv_key = dict(map(reversed, self.key.items()))

                if what == "bin":
                    # Create a binary matrix
                    res_bin = np.zeros((len(inputs), len(self.key)))
                    # Iterate through the processed data and fill the binary matrix
                    for i, item in enumerate(processed_data):
                        for sub_item in item:
                            if sub_item in inv_key:
                                res_bin[i, inv_key[sub_item]] = 1
                    
                    res = res_bin
                    res = pd.DataFrame(res, columns=[f"{self.system}_{self.key[i]}" for i in range(len(self.key))])

                elif what == "pred":

                    res = []
                    # Insert description
                    for item in processed_data:
                        codes = []
                        for sub_item in item:
                            try:
                                if sub_item in inv_key:
                                    codes.append(inv_key[sub_item])
                                else:
                                    codes.append(f'u{sub_item}')  # Add 'u' to ensure being able to pick it up in cleaning below
                            except (ValueError, TypeError):
                                # Handle the case where sub_item cannot be cast to float
                                codes.append(f'u{sub_item}')  # Add 'u' to ensure being able to pick it up in cleaning below

                        row = [[self.key[i], self.key_desc[i]] if i in self.key else [i[1:], "Unknown code"] for i in codes]
                        row = [item for sublist in row for item in sublist] # Flatten list
                        res.append(row)

                    column_names = []
                    for i in range(1, max_elements+1):column_names.extend([f'{self.system}_{i}', f'desc_{i}'])

                    # Create the DataFrame
                    res = pd.DataFrame(res, columns=column_names)

                    # Identify columns starting with 'prob_s2s_'
                    prob_cols = [col for col in out.columns if col.startswith('prob_s2s_')]

                    # Multiply these columns row-wise
                    res['conf'] = out[prob_cols].prod(axis=1)
                    if order_invariant_conf:
                        # Use order invariant if it is >0
                        order_inv_conf = out.order_inv_conf.fillna(0)
                        res['conf'] = np.where(order_inv_conf > 0, order_inv_conf, res['conf'])

                    # Add input
                    res.insert(0, 'occ1', out.input)

                return res
            
            else:
                raise ValueError(f"'what' ('{what}') did not match any output for 'out_type' ('{out_type}')")

        return res

    def _split_str_s2s(self, pred: str, symbol: str = "&"):
        """
        Splits predicted str if necessary
        """
        if symbol in pred:
            pred = pred.split(symbol)

        return pred

    def finetune(
            self,
            dataset: str | os.PathLike,
            save_path: str | os.PathLike,
            input_col: str,
            language: str,
            language_col: str | None,
            save_interval: int,
            log_interval: int,
            eval_interval: int,
            drop_bad_labels: bool,
            allow_codes_shorter_than_block_size: bool,
            share_val: float,
            learning_rate: float,
            num_epochs: int,
            warmup_steps: int,
            seq2seq_weight: float,
            freeze_encoder: bool,
            ):

        # Data prep
        prepare_finetuning_data( # TODO this will save a keys-file, which is NOT the one we'll be using
            dataset=dataset,
            input_col=input_col,
            formatter=self.formatter,
            save_path=save_path,
            share_val=share_val,
            language=language,
            language_col=language_col,
            drop_bad_rows=drop_bad_labels,
            allow_codes_shorter_than_block_size=allow_codes_shorter_than_block_size,
        )

        # Load datasets
        dataset_train, dataset_val = setup_finetuning_datasets(
            target_cols=self.formatter.target_cols,
            save_path=save_path,
            formatter=self.formatter,
            tokenizer=self.tokenizer,
            num_classes_flat=len(self.key),
            map_code_label={v: k for k, v in self.key.items()}, # We need the reverse mapping to produce IDXs from codes
        )

        # Data loaders
        data_loader_train = DataLoader(
            dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            )
        data_loader_val = DataLoader(
            dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            )

        if freeze_encoder:
            for param in self.model.encoder.parameters():
                param.requires_grad = False

            optimizer = AdamW([param for name, param in self.model.named_parameters() if not name.startswith("encoder.")], lr=learning_rate)
        else:
            optimizer = AdamW(self.model.parameters(), lr=learning_rate)

        total_steps = len(data_loader_train) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # Setup mixed loss
        loss_fn_seq2seq = BlockOrderInvariantLoss(
            pad_idx=PAD_IDX,
            nb_blocks=self.formatter.max_num_codes,
            block_size=self.formatter.block_size,
        )
        loss_fn_linear = torch.nn.BCEWithLogitsLoss()
        loss_fn = LossMixer(
            loss_fn_seq2seq=loss_fn_seq2seq,
            loss_fn_linear=loss_fn_linear,
            seq2seq_weight=seq2seq_weight,
        ).to(self.device)

        # Load states
        current_step = load_states(
            save_dir=save_path,
            model=self.model,
            optimizer=optimizer,
            scheduler=scheduler,
        )

        train(
            model=self.model,
            data_loaders={
                'data_loader_train': data_loader_train,
                'data_loader_val': data_loader_val,
            },
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=self.device,
            scheduler=scheduler,
            save_dir=save_path,
            total_steps=total_steps,
            current_step=current_step,
            log_interval=log_interval,
            eval_interval=eval_interval,
            save_interval=save_interval
        )
