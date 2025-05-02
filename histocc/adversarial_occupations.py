# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 13:56:27 2024

@author: christian-vs

This script implements tools to use for finding strings describing occupations,
which are hard for OccCANINE to define.

Criteria for adversarial occupation:
    1. Tranlastable: Must be able to translate string into English
    2. Valid label: Det correct label must be predicted
"""

import math
import os
import random
import string
import sys
import time
from unidecode import unidecode

import torch

from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

import pandas as pd

from .prediction_assets import OccCANINE
from .attacker import AttackerClass


# Lang. abbreviation mappings
lang_mapping = { # OccCANINE --> facebook/m2m100_418M: Table A1 --> Table 1
    'ca': 'ca',  # Catalan
    'da': 'da',  # Danish
    'ge': 'de',  # German
    'en': 'en',  # English
    'es': 'es',  # Spanish
    'fr': 'fr',  # French
    'gr': 'el',  # Greek
    'is': 'is',  # Icelandic
    'it': 'it',  # Italian
    'nl': 'nl',  # Dutch
    'no': 'no',  # Norwegian
    'pt': 'pt',  # Portuguese
    'se': 'sv',  # Swedish
    'unk': 'unk', # Unknown
}

class Translator:
    """
    A class to handle translation tasks using the facebook/m2m100_418M model.
    """

    # Paper: https://arxiv.org/pdf/2010.11125

    def __init__(self, device: str | None = None):
        """
        Initializes the Translator class by loading the model and tokenizer, and setting the device.

        Parameters:
        device (str, optional): The device to run the model on ('cpu' or 'cuda'). Defaults to None.
        """
        self.model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
        self.tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

        # Translate Danish to English
        self.tokenizer.src_lang = "da"
        self.lang_to = self.tokenizer.get_lang_id("en")

        # Detect device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)
        print(f"Using device: {self.device}")
        self.model.to(self.device)

        # Lang. abbreviation mappings
        self.lang_mapping = lang_mapping

    def translate(self, text: str | list[str], lang_from: str, lang_to: str) -> list[str]:
        """
        Translates a given text from one language to another.

        Parameters:
        text (str or list of str): The text to translate.
        lang_from (str): The source language code.
        lang_to (str): The target language code.

        Returns:
        list of str: The translated text.
        """
        # Turn into valid langs for m2m100_418M
        lang_from = self._convert_lang_abr(lang_from)
        lang_to = self._convert_lang_abr(lang_to)

        # Handle 'unk' language
        if lang_from=='unk' or lang_to=='unk':
            raise NotImplementedError("A way of handling unknown languages is not implemented yet")

        if isinstance(text, str):
            text = [text]

        if not isinstance(text, list):
            raise TypeError("Should be 'list' or 'str'")

        # Convert language abbreviations if necessary
        lang_from = self._convert_lang_abr(lang_from)
        lang_to = self._convert_lang_abr(lang_to)

        # Set language from
        self.tokenizer.src_lang = lang_from

        # Set language to
        target_lang = self.tokenizer.get_lang_id(lang_to)

        encoded = [self.tokenizer(x, return_tensors="pt") for x in text]
        encoded = [{k: v.to(self.device) for k, v in e.items()} for e in encoded]
        generated_tokens = [self.model.generate(**x, forced_bos_token_id=target_lang) for x in encoded]
        res = [self.tokenizer.batch_decode(x, skip_special_tokens=True) for x in generated_tokens]

        # encoded = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        # generated_tokens = self.model.generate(**encoded, forced_bos_token_id=target_lang)
        # res = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        # Tolower
        res = [x[0].lower() for x in res]

        return res

    def double_translate(self, text: str | list[str], lang_from: str, lang_to: str) -> list[str]:
        """
        Translates a text from one language to another and then back to the original language.

        Parameters:
        text (str or list of str): The text to translate.
        lang_from (str): The source language code.
        lang_to (str): The intermediate target language code.

        Returns:
        list of str: The doubly translated text.
        """
        step1 = self.translate(text, lang_from=lang_from, lang_to=lang_to)
        step2 = self.translate(step1, lang_from=lang_to, lang_to=lang_from)

        return step2

    def _convert_lang_abr(self, lang: str) -> str:
        """
        Converts a language abbreviation from OccCANINE to the abbreviation used in facebook/m2m100_418M.

        Parameters:
        lang (str): The language abbreviation to convert.

        Returns:
        str: The converted language abbreviation.
        """
        return self.lang_mapping.get(lang, lang)  # Default to returning the same abbreviation if not found


class HiddenPrints: # https://stackoverflow.com/a/45669280
    """
    A context manager to suppress print statements.
    """
    _original_stdout = sys.stdout

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w', encoding='utf-8')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


# Function to generate ETA string
def eta(i, start_time: float, cap_n: int) -> str:
    """
    Generates an ETA string based on the current progress.

    Parameters:
    i (int): Current iteration index.
    start_time (float): The start time of the process.
    capN (int): Total number of iterations.

    Returns:
    str: The ETA string.
    """
    elapsed_time = time.time() - start_time
    average_time_per_n = elapsed_time / (i+1)
    remaining_n = cap_n - (i+1)
    eta_seconds = remaining_n * average_time_per_n

    # Convert eta_seconds to hours, minutes, and seconds
    eta_hours = int(eta_seconds // 3600)
    eta_minutes = int((eta_seconds % 3600) // 60)
    eta_seconds = int(eta_seconds % 60)
    eta_str = f"{eta_hours}h {eta_minutes}m {eta_seconds}s"

    total_seconds = cap_n * average_time_per_n

    # Convert total_seconds to hours, minutes, and seconds
    total_hours = int(total_seconds // 3600)
    total_minutes = int((total_seconds % 3600) // 60)
    total_seconds = int(total_seconds % 60)
    total_str = f"{total_hours}h {total_minutes}m {total_seconds}s"

    eta_str = f"{i} of {cap_n} {eta_str} of {total_str}"

    return eta_str


class AdversarialStrings:
    """
    This class implements everything for finding adversial strings
    """

    lang: str = None

    def __init__(
            self,
            attacker: AttackerClass,
            translator: Translator,
            predictor: OccCANINE,
            df: pd.DataFrame,
    ):
        """
        Initializes the AdversarialStrings class.

        Parameters:
        attacker: The attacker instance.
        translator: The translator instance.
        predictor: The predictor instance.
        df (DataFrame): DataFrame containing 'occ1'
        """
        self.attacker = attacker
        self.translator = translator
        self.predictor = predictor
        self.n = 0

        # Run OccCANINE prediction for all labels at setup
        self.pred0 = self._run_occ_canine_predicts(df.occ1.tolist(), df.lang.tolist())

    def _goalfunction(self, pred0_bin, aug_text):
        """
        Decides when an augmentation has been successful. Based on the binary
        representation, which is indifferent to order.

        Parameters:
        pred0: Original prediction.
        aug_text: Augmented text.

        Returns:
        bool: True if the augmented prediction is different from the original.
        """
        pred_aug = self.predictor.predict(aug_text, lang = self.lang, what = "bin")

        if len(pred_aug) != 1:
            raise ValueError(f"Somehow more than one occupational description was passed to OccCANINE.predict(); Input: {aug_text}")

        # Change both to list
        pred0_bin = [self._isnan_to_str(x) for x in pred0_bin.iloc[0].tolist()]
        pred_aug = [self._isnan_to_str(x) for x in pred_aug.iloc[0].tolist()]

        # Equality check

        return pred_aug != pred0_bin

    def _isnan_to_str(self, x):
        if isinstance(x, str):
            if x == ' ':
                return x
            
            try: # If we can conver tit to numerical (most often we can), then its preferable
                x = float(x)
            except ValueError:
                pass

            
            return x
        else:
            if math.isnan(x):
                return ' '
            else:
                return x

    def _run_occ_canine_predicts(self, text, lang):
        # To add: Classify -1 and "" as the same
        # with HiddenPrints():
        self.predictor.verbose = True
        print("Running predictions to validate augmentable strings:")
        pred0 = self.predictor.predict(text, lang = lang)
        print("--> Finished predictions")
        self.predictor.verbose = False

        system = self.predictor.system

        res = []
        for _, row in pred0.iterrows():
            res_i = [row[f'{system}_{i+1}'] for i in range(5) if f'{system}_{i+1}' in row]
            res_i = [self._isnan_to_str(x) for x in res_i]
            res.append(res_i)

        return res

    def _validate_augmentation(self, text, labels, i):
        """
        Validates if the given text is appropriate for augmentation.

        Parameters:
        text (str): The text to validate.
        labels (list): The expected labels.

        Returns:
        bool: True if the text passes validation tests.
        """
        # Handle 'unk' language
        if self.lang=='unk':
            return False

        # Handle 'unk' language
        if self.lang=='en':
            lang_to = 'fr'
        else:
            lang_to = 'en'

        # Test 1: Can be translated
        translation = self.translator.translate(text, lang_from=self.lang, lang_to=lang_to)[0]

        test1 = text != translation

        # Test 2: Correct initial prediction

        labels = [self._isnan_to_str(x) for x in labels]
        pred0 = [self._isnan_to_str(x) for x in self.pred0[i]]

        test2 = all([x==y for x, y in zip(pred0, labels)])

        return test1 and test2

    def get_adv_string(self, text, lang, labels, i, verbose_extra = False, n_max = 100, double_translate = True):
        """
        Runs adversarial attack on a piece of text.

        Parameters:
        text (str): The text to attack.
        lang (str): The language of the text.
        labels (list): The expected labels.
        verbose_extra (bool): If True, prints additional information.
        n_max (int): Maximum number of augmentation attempts.
        double_translate (bool): If True, double translation is used as the primary augmentation.


        Returns:
        tuple: The adversarial text and the number of attempts.
        """
        self.lang = lang
        self.lang_to = 'en'

        if lang == 'en':
            self.lang_to = 'fr' # Translate through french if input is 'en'

        # Validate augmentation
        test = self._validate_augmentation(text, labels, i)
        if not test:
            return None

        # Predict HISCO codes from raw string
        pred0_bin = self.predictor.predict(text, lang = lang, what = "bin") # Legacy name, not updated
        aug_text = text

        occs = [text] # Store all the strings we have tried to augment

        for i in range(n_max):
            if double_translate:
                aug_text1 = self.translator.double_translate(aug_text, self.lang, lang_to=self.lang_to)[0]
                if aug_text1 == aug_text: # Then it is stuck in a loop
                    aug_text1 = self.attacker.attack(aug_text)
                aug_text = aug_text1
            else:
                aug_text = self.attacker.attack(aug_text) # Arguments in construction of class

            attack_success = self._goalfunction(pred0_bin, aug_text)

            occs.append(aug_text)

            if attack_success:
                if verbose_extra:
                    print(f"Succeful attack after {i+1} augments:\n{text} --> {aug_text}")

                    pred = self.predictor.predict(text, lang = lang, get_dict = True)
                    print("")
                    print("Initial label prediction:")
                    print(pred)

                    print("")
                    print("Attacked label prediction:")
                    pred = self.predictor.predict(aug_text, lang = lang, get_dict = True)
                    print(pred)
                    print("")

                break

        return occs


def generate_advanced_gibberish(min_words = 1, max_words = 10, min_length = 1, max_length = 10, punctuation = 0.5):
    """
    Generates a string of random words with specified constraints on the number of words
    and their lengths.

    Parameters:
    min_words (int): Minimum number of words in the generated string.
    max_words (int): Maximum number of words in the generated string.
    min_length (int): Minimum length of each word.
    max_length (int): Maximum length of each word.
    punctuation (float): Probability of including punctuation in the generated string.

    Returns:
    str: A string consisting of random words separated by spaces.
    """
    words = random.randint(min_words, max_words)
    word_lengths = [random.randint(min_length, max_length) for _ in range(words)]

    # Lowercase
    string_set = string.ascii_lowercase

    # Flip coin to decide if we want to include use punctuation or not
    if random.random() < punctuation:
        string_set += string.ascii_lowercase + string.ascii_lowercase + string.punctuation + string.digits

    words = [''.join(random.choices(string_set, k=x)) for x in word_lengths]

    sentence = ' '.join(words)

    return sentence


def generate_random_strings(num_strings, system = 'hisco', no_occ_labels = [-1], langs = lang_mapping.keys()):
    """
    Generates random strings and assigns a random valid language.

    Parameters:
    num_strings (int): Number of random strings to generate.

    Returns:
    pd.DataFrame: DataFrame containing random strings, label -1, and a random valid language.
    """
    random_strings = [generate_advanced_gibberish() for _ in range(num_strings)]

    valid_languages = [lang for lang in langs]
    random_langs = [random.choice(valid_languages) for _ in range(num_strings)]

    no_occ_label = [random.choice(no_occ_labels) for _ in range(num_strings)]

    result = pd.DataFrame(
        {'occ1': random_strings,
         f'{system}_1': no_occ_label,
         f'{system}_2': [' ']*num_strings,
         f'{system}_3': [' ']*num_strings,
         f'{system}_4': [' ']*num_strings,
         f'{system}_5': [' ']*num_strings,
         'lang': random_langs
        })

    return result


def load_training_data(data_path = "Data/Training_data", toyload=False, verbose = True, sample_size = None, return_lang_counts = False):
    """
    Loads and combines training data from CSV files in the specified folder.

    Parameters:
    data_path (str): The folder containing the CSV files.
    toyload (bool): If True, loads only a small subset of data for testing.
    verbose (bool): If True, prints progress information.
    sample_size (float): Number of observations to read. Defaults to reading all
    return_lang_countsreturn_lang_counts (bool): Should each language's proportion be estimated and returned. (Used in 'translated_strings_wrapper')

    Returns:
    tuple: The combined dataframe of training data and a dictionary with lang. counts (if return_lang_counts is True).
    """
    if os.path.isfile(data_path) and data_path.endswith(".csv"):
        # If the provided path is a single CSV file
        df = pd.read_csv(data_path)
        df = df.drop(columns=['RowID'], errors='ignore')
        df = df.drop_duplicates()
        df["source"] = os.path.basename(data_path)

        # Update language counts if required
        lang_counts = {}
        if return_lang_counts:
            for lang, count in df['lang'].value_counts().items():
                lang_counts[lang] = count

        # Handle sampling if sample_size is provided
        if sample_size:
            actual_n = min(sample_size, df.shape[0])
            df = df.sample(n=actual_n, replace=False)

        # Ensure all occ1 are strings
        df["occ1"] = df["occ1"].astype(str)

        if return_lang_counts:
            # Aggregate language counts into proportions
            total_count = sum(lang_counts.values())
            lang_counts = {lang: int(sample_size * count / total_count) for lang, count in lang_counts.items()}
            return df, lang_counts

        return df

    else:
        # If the provided path is a directory containing CSV files
        fnames = os.listdir(data_path)

        if toyload:
            fnames = fnames[0:2] + [fnames[5]]

        if sample_size:
            share_of_sample = sample_size // len(fnames)
            if share_of_sample < 1:
                share_of_sample = 1

        # Initialize an empty dataframe to store the data
        combined_df = pd.DataFrame()

        lang_counts = {}

        # Loop through the file list and read each CSV file into the combined dataframe
        for file in fnames:
            if file.endswith(".csv"):  # Make sure the file is a CSV file
                file_path = os.path.join(data_path, file)

                if toyload:
                    df = pd.read_csv(file_path, nrows=25)
                else:
                    df = pd.read_csv(file_path)

                df = df.drop(columns=['RowID'], errors='ignore')
                df = df.drop_duplicates()
                df["source"] = file

                # Update language counts
                if return_lang_counts:
                    for lang, count in df['lang'].value_counts().items():
                        if lang in lang_counts:
                            lang_counts[lang] += count
                        else:
                            lang_counts[lang] = count

                # Handling 'share_of_sample' larger than samples in n
                actual_n = df.shape[0] if share_of_sample > df.shape[0] else share_of_sample

                df = df.sample(n=actual_n, replace=False)

                combined_df = pd.concat([combined_df, df])

                n_df = df.shape[0]

                if verbose:
                    print(f"\nRead {file} (N = {n_df})")

        df = combined_df

        # Make sure that all occ1 are strings
        df["occ1"] = df["occ1"].astype(str)

        if return_lang_counts:
            # Aggregate language counts into proportions
            total_count = sum(lang_counts.values())
            lang_counts = {lang: int(sample_size * count / total_count) for lang, count in lang_counts.items()}
            return df, lang_counts

        return df
    

def balance_classes(df, system = 'hisco'):
    """
    Balances the classes in the DataFrame by sampling from each class.

    Parameters:
    df (DataFrame): The DataFrame to balance.
    system (str): The system to use for balancing.
    n_samples (int): The number of samples to take from each class.

    Returns:
    DataFrame: The balanced DataFrame.
    """
    # Drop index
    df = df.reset_index(drop=True)

    # Try converting all columns to float (if e.g. PST then this will not work and pass)
    for i in range(1, 6):
        column_name = f'{system}_{i}'
        if column_name in df.columns:
            df[column_name] = df[column_name].astype(str)
            try: 
                df[column_name] = df[column_name].astype(float)
            except ValueError:
                try:
                    # Replace " " with NaN and convert to float
                    df[column_name] = df[column_name].replace(" ", float('nan'))
                    df[column_name] = df[column_name].astype(float)
                except ValueError:
                    pass

    # Get the counts of each class
    unique_classes = []
    for i in range(1, 6):
        column_name = f'{system}_{i}'
        if column_name in df.columns:
            unique_classes.extend(df[column_name].unique())

    # Convert to float if possible (not crucial - might oversample a bit based on columns loaded as different types)
    for i in range(len(unique_classes)):
        try:
            unique_classes[i] = float(unique_classes[i])
        except ValueError:
            pass
    
    unique_classes = list(set(unique_classes))

    # Drop nan
    unique_classes = [x for x in unique_classes if str(x) != 'nan']
    
    # N samples as fixed share of df size
    n_samples = df.shape[0] // len(unique_classes)
    if n_samples < 1:
        n_samples = 1

    # Sample from each class
    balanced_df = pd.DataFrame()
    for cls in unique_classes:
        
        # Which rows contain at least one of the classes across all columns?
        class_indices = []
        for i in range(1, 6):
            column_name = f'{system}_{i}'
            if column_name in df.columns:
                class_indices.append(df[df[column_name] == cls].index.tolist()) 

        # Flatten list
        class_indices = list(set([item for sublist in class_indices for item in sublist]))
            
        relevant_df = df.iloc[class_indices]
        
        sampled_class_df = relevant_df.sample(n=n_samples, replace=True)
        balanced_df = pd.concat([balanced_df, sampled_class_df])

    return balanced_df


def generate_adversarial_wrapper(
        data_path, hisco_predictor, 
        toyload=False, double_translate = True, sample_size = 1000, n_max = 10,
        verbose=True, verbose_extra=False, alt_prob = 1, n_trans=1, class_balance = True
        ):
    """
    Main function to generate adversarial examples.

    Parameters:
    data_path (str): Which folder can the data be found it? The function will load all the .csv files in this destination
    hisco_predictor (OccCANINE): The OccCANINE model to use for predictions.
    toyload (bool): If True, loads only a small subset of data for testing.
    double_translate (bool): Should double translation be the primary augmentation? Otherwise it is just attacker.attack()
    sample_size (int): Number of observations to use
    n_max (int): Maximum number of augmentation attempts.
    verbose (bool): If True, prints progress information.
    verbose_extra (bool): If True, prints additional information.
    alt_prob (float): Probability of using alterations.
    n_trans (int): Number of double translations to perform per iteration.
    class_balance (bool): If True, balances the classes in the DataFrame.

    This function:
    1. Loads training data.
    2. Initializes translator, attacker, and predictor classes.
    3. Iterates over each row in the dataset, attempting to find adversarial examples.
    4. Saves the results to a CSV file.
    """

    # System
    system = hisco_predictor.system

    # Load data
    df = load_training_data(data_path = data_path, toyload=toyload, sample_size=sample_size)
    df = df.drop_duplicates()
    if toyload: df = df[0:20]

    if class_balance:
        df = balance_classes(df, system = system)
    
    # Assign df id
    df = df.reset_index(drop = True)
    df['aug_id'] = df.index    

    # Print update
    if verbose:
        n_df = df.shape[0]
        print(f"\nRead everything (N = {n_df})")

    # Init translator
    translator = Translator()

    # Init attacker class
    attacker = AttackerClass(alt_prob = alt_prob, n_trans=n_trans, df=df)

    # Class to handle adverarial string finding
    adv_strings = AdversarialStrings(attacker, translator, hisco_predictor, df)

    # For ETA print
    start_time = time.time()
    cap_n = len(df)

    # Apply adversarial string finding to each row
    results = pd.DataFrame()
    i = 0
    for _, row in df.iterrows():
        text = row['occ1']
        lang = row['lang']
        labels = [
            row[f'{system}_1'],
            row[f'{system}_2'],
            row[f'{system}_3'],
            row[f'{system}_4'],
            row[f'{system}_5']
        ]
        res_i = adv_strings.get_adv_string(
            text, lang, labels, verbose_extra=verbose_extra, i=i, n_max=n_max, double_translate=double_translate
            )
        if res_i:
            if verbose:
                print(f"--> Found {len(res_i)} adversarial examples for {text}")
                print(f"{eta(i=i, start_time=start_time, cap_n=cap_n)} --> Stopped after {len(res_i)} of {n_max}")

            res_i = pd.DataFrame({'occ1': res_i}) # Convert to DataFrame
        
            # Add in labels
            res_i[f'{system}_1'] = [row[f'{system}_1'] for _ in range(len(res_i))]
            res_i[f'{system}_2'] = [row[f'{system}_2'] for _ in range(len(res_i))]  
            res_i[f'{system}_3'] = [row[f'{system}_3'] for _ in range(len(res_i))]
            res_i[f'{system}_4'] = [row[f'{system}_4'] for _ in range(len(res_i))]
            res_i[f'{system}_5'] = [row[f'{system}_5'] for _ in range(len(res_i))]
            res_i['aug_id'] = [row['aug_id'] for _ in range(len(res_i))] # Unique ID for each agumented string 
            res_i['n_aug'] = [j for j in range(len(res_i))] # Number of augmentations

            results = pd.concat([results, res_i])

        i += 1

    # Merge on aug_id
    results = results.reset_index(drop = True)
    df = df.reset_index(drop = True)
    # delelte info found in results
    df = df.drop(columns=['occ1',f'{system}_1', f'{system}_2', f'{system}_3', f'{system}_4', f'{system}_5'])
    results = results.merge(df, on='aug_id', how='left')

    return results


def translated_strings_wrapper(data_path, toyload=False, verbose=True, sample_size = 1000):
    """
    Translates occupation strings in a dataset to multiple languages using a predefined translator.

    Parameters:
    data_path (str): Which folder can the data be found it? The function will load all the .csv files in this destination
    toyload (bool, optional): Determines whether to load a smaller toy dataset or the full dataset. Defaults to True.
    verbose (bool, optional): If True, prints progress updates during the translation process. Defaults to True.

    Returns:
    pd.DataFrame: A DataFrame containing the translated occupation strings.

    The function performs the following steps:
    1. Initializes the translator.
    2. Loads the training data, either a toy dataset or the full dataset based on the `toyload` parameter.
    3. Removes duplicate entries and filters out rows where the language is unknown ('unk').
    4. Iterates over each language in the translator's language mapping.
    5. Translates each occupation string from its original language to the target language.
    7. Concatenates the results into a single DataFrame.
    8. Saves the DataFrame containing the translated strings.
    """

    # Init translator
    translator = Translator()

    df, lang_counts = load_training_data(data_path = data_path, toyload=toyload, sample_size = sample_size, return_lang_counts=True)
    df = df.drop_duplicates()
    df = df[df['lang'] != 'unk']

    n_df = df.shape[0]
    print(f"\nRead everything (N = {n_df})")
    print()

    # More than one lang
    only_one_lang = len(lang_counts) == 1
    if only_one_lang:
        if verbose:
            print("Only one language in the dataset.")

        n_each_lang = n_df // len(lang_mapping) # Number of observations to translate into each language
        if n_each_lang < 1:
            n_each_lang = 1

        lang_counts = {lang: n_each_lang for lang in lang_mapping.keys()}  # Set all to same share

    if verbose:
        print("The following number of translations will be produced*:")
        n_total = sum(lang_counts.values())
        for lang in lang_counts:
            n_lang = lang_counts[lang]
            pct_lang = n_lang / n_total
            print(f" --> {n_lang} will be translated into '{lang}' -- {pct_lang:.2%}")

        if only_one_lang:
            print("   *This is equal proportions of all the available languages")
        else:
            print("   *This reflects proportions in the full traning data")

    results = pd.DataFrame()
    for lang in lang_counts:
        if lang == 'unk':
            continue
        df_lang = df[df['lang'] != lang]  # Only translate those that are not already in the target language

        if df_lang.shape[0] == 0:
            if verbose:
                print(f"--> No observations to translate to '{lang}'")
            continue

        # Sample df_lang to be of size given by lang_proportions
        n_lang = lang_counts[lang]
        actual_n = df_lang.shape[0] if n_lang > df_lang.shape[0] else n_lang # Handling cases with too large n

        df_lang = df_lang.sample(actual_n)

        if n_lang == 0:
            print(f"--> Finished translating {n_lang} observations to '{lang}'")
            continue

        # PERFORM TRANSLATION:
        translated_occ = translator.translate(df_lang['occ1'].tolist(), lang_from=df_lang['lang'].tolist()[0], lang_to=lang)

        if lang == 'gr':
            # Replace greek letters with english letters
            translated_occ = [unidecode(x) for x in translated_occ]

        if verbose:
            print(f"--> Finished translating {n_lang} observations to '{lang}'")

        res_i = pd.DataFrame({'translated_occ': translated_occ}) # Convert to DataFrame
        res_i.reset_index(drop=True, inplace=True)
        df_lang.reset_index(drop=True, inplace=True)
        res_i = pd.concat([df_lang, res_i], axis=1)

        # set lang
        res_i['original_lang'] = res_i['lang']
        res_i['lang'] = [lang for i in range(len(res_i))]

        results = pd.concat([results, res_i])

    # Make 'occ1' the augmented strings - store original string as 'original_occ1'
    original_occ1 = results['occ1'].tolist()
    translated_occ = results['translated_occ'].tolist()
    results['occ1'] = translated_occ
    results['original_occ1'] = original_occ1
    results = results.drop(columns='translated_occ')

    return results


def generate_random_strings_wrapper(num_strings=1000, system = 'hisco', no_occ_labels = [-1], langs = lang_mapping.keys()):
    """
    Wrapper function to generate random strings with the label -1 and save them.

    Parameters:
    num_strings (int, optional): Number of random strings to generate. Defaults to 1000.
    system (str, optional): The system to use for the generated strings. Defaults to 'hisco'.
    no_occ_labels (list of str or int): Labels for the generated strings. Should be the labels covering 'no occupation'. Can be more than one. Defaults to [-1] (HISCO).
    langs (list, optional): List of languages to use for the generated strings. Defaults to all available languages.

    Returns:
    None
    """
    random_strings_df = generate_random_strings(num_strings, system = system, no_occ_labels = no_occ_labels, langs = langs)

    return random_strings_df
