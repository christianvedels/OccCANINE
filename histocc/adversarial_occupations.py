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
        
        pred_aug = pred_aug[0].tolist()
        return pred_aug != pred0_bin

    def _isnan_to_str(self, x):
        if isinstance(x, str):
            if x == ' ':
                return x
            x = float(x)
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

        res = []
        for _, row in pred0.iterrows():
            res_i = [
                row['hisco_1'],
                row['hisco_2'],
                row['hisco_3'],
                row['hisco_4'],
                row['hisco_5']
                ]
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

        test2 = all([x==y for x, y in zip(self.pred0[i], labels)])

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
            return None, -1

        # Predict HISCO codes from raw string
        pred0_bin = self.predictor.predict(text, lang = lang, what = "bin")[0].tolist()
        aug_text = text

        for i in range(n_max):
            if double_translate:
                aug_text1 = self.translator.double_translate(aug_text, self.lang, lang_to=self.lang_to)[0]
                if aug_text1 == aug_text: # Then it is stuck in a loop
                    aug_text1 = self.attacker.attack(aug_text)
                aug_text = aug_text1
            else:
                aug_text = self.attacker.attack(aug_text) # Arguments in construction of class

            if self._goalfunction(pred0_bin, aug_text):
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

        return aug_text, (i+1)


def generate_advanced_gibberish(min_words = 1, max_words = 10, min_length = 1, max_length = 10):
    """
    Generates a string of random words with specified constraints on the number of words
    and their lengths.

    Parameters:
    min_words (int): Minimum number of words in the generated string.
    max_words (int): Maximum number of words in the generated string.
    min_length (int): Minimum length of each word.
    max_length (int): Maximum length of each word.

    Returns:
    str: A string consisting of random words separated by spaces.
    """
    words = random.randint(min_words, max_words)
    word_lengths = [random.randint(min_length, max_length) for _ in range(words)]

    words = [''.join(random.choices(string.ascii_lowercase, k=x)) for x in word_lengths]

    sentence = ' '.join(words)

    return sentence


def generate_random_strings(num_strings):
    """
    Generates random strings and assigns a random valid language.

    Parameters:
    num_strings (int): Number of random strings to generate.

    Returns:
    pd.DataFrame: DataFrame containing random strings, label -1, and a random valid language.
    """
    random_strings = [generate_advanced_gibberish() for _ in range(num_strings)]

    valid_languages = [lang for lang in lang_mapping.keys() if lang != 'unk']
    random_langs = [random.choice(valid_languages) for _ in range(num_strings)]

    return pd.DataFrame(
        {'occ1': random_strings,
         'hisco_1': [-1]*num_strings,
         'hisco_2': [' ']*num_strings,
         'hisco_3': [' ']*num_strings,
         'hisco_4': [' ']*num_strings,
         'hisco_5': [' ']*num_strings,
         'code1': [2]*num_strings,
         'code2': [' ']*num_strings,
         'code3': [' ']*num_strings,
         'code4': [' ']*num_strings,
         'code5': [' ']*num_strings,
         'lang': random_langs
        })


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
    fnames = os.listdir(data_path)

    if toyload:
        fnames = fnames[0:2]+[fnames[5]]
    
    if sample_size:
        share_of_sample = sample_size // len(fnames)

    # Initialize an empty dataframe to store the data
    combined_df = pd.DataFrame()
    
    lang_counts = {}
        
    # Loop through the file list and read each CSV file into the combined dataframe
    for file in fnames:
        if file.endswith(".csv"):  # Make sure the file is a CSV file
            file_path = os.path.join(data_path, file)

            if toyload:
                df = pd.read_csv(file_path, nrows = 25)
            else:
                df = pd.read_csv(file_path)

            df = df.drop(columns=['RowID'])
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

            df = df.sample(n = actual_n, replace=False)

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


def generate_adversarial_wrapper(
        data_path, storage_path,
        toyload=False, double_translate = True, sample_size = 1000, n_max = 10, 
        verbose=True, verbose_extra=False, alt_prob = 1, n_trans=1
        ):
    """
    Main function to generate adversarial examples.

    Parameters:
    data_path (str): Which folder can the data be found it? The function will load all the .csv files in this destination
    storage_path (str): Where should the results be stored?
    toyload (bool): If True, loads only a small subset of data for testing.
    double_translate (bool): Should double translation be the primary augmentation? Otherwise it is just attacker.attack()
    sample_size (int): Number of observations to use
    n_max (int): Maximum number of augmentation attempts.


    This function:
    1. Loads training data.
    2. Initializes translator, attacker, and predictor classes.
    3. Iterates over each row in the dataset, attempting to find adversarial examples.
    4. Saves the results to a CSV file.
    """

    df = load_training_data(data_path = data_path, toyload=toyload, sample_size=sample_size)
    df = df.drop_duplicates()

    n_df = df.shape[0]
    print(f"\nRead everything (N = {n_df})")

    # Init translator
    translator = Translator()

    # Init attacker class
    attacker = AttackerClass(alt_prob = alt_prob, n_trans=n_trans, df=df)

    # Predictor
    hisco_predictor = OccCANINE(name = "CANINE")

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
            row['hisco_1'],
            row['hisco_2'],
            row['hisco_3'],
            row['hisco_4'],
            row['hisco_5']
        ]
        res_i = adv_strings.get_adv_string(
            text, lang, labels, verbose_extra=verbose_extra, i=i, n_max=n_max, double_translate=double_translate
            )
        n_augs = res_i[1]
        res_i = pd.DataFrame({'aug_string': [res_i[0]], 'attempts': [res_i[1]]}) # Convert to DataFrame
        results = pd.concat([results, res_i])
        
        if verbose:
            print(f"{eta(i=i, start_time=start_time, cap_n=cap_n)} --> Stopped after {n_augs} of {n_max}")
        i += 1

    # Reset the index of the results DataFrame if necessary
    results.reset_index(drop=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    results = pd.concat([df, results], axis=1)
    
    # Make 'occ1' the augmented strings - store original string as 'original_occ1'
    original_occ1 = results['occ1'].tolist()
    aug_string = results['aug_string'].tolist()
    results['occ1'] = aug_string
    results['original_occ1'] = original_occ1

    results = results[results['occ1'].notna()]
    
    results = results.drop(columns='aug_string')
    
    # Save results
    fname = os.path.join(storage_path, f"Adv_data_double_translate{double_translate}.csv")
    results.to_csv(fname, index = False)


def translated_strings_wrapper(data_path, storage_path, toyload=False, verbose=True, sample_size = 1000):
    """
    Translates occupation strings in a dataset to multiple languages using a predefined translator.

    Parameters:
    data_path (str): Which folder can the data be found it? The function will load all the .csv files in this destination
    storage_path (str): Where should the results be stored?
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
    
    if verbose:
        print("The following number of translations will be produced*:")
        n_total = sum(lang_counts.values())
        for lang in lang_counts:
            n_lang = lang_counts[lang]
            pct_lang = n_lang / n_total
            print(f" --> {n_lang} will be translated into '{lang}' -- {pct_lang:.2%}")
        
        print("   *This reflects proportions in the full traning data")
        

    results = pd.DataFrame()
    for lang in lang_counts:
        if lang == 'unk':
            continue
        df_lang = df[df['lang'] != lang]  # Only translate those that are not already in the target language
        
        # Sample df_lang to be of size given by lang_proportions
        n_lang = lang_counts[lang]
        actual_n = df_lang.shape[0] if n_lang > df_lang.shape[0] else n_lang # Handling cases with too large n
        
        df_lang = df_lang.sample(actual_n)
        
        if n_lang == 0:
            print(f"--> Finished translating {n_lang} observations to '{lang}'")
            continue
        
        # PERFORM TRANSLATION:
        translated_occ = translator.translate(df_lang['occ1'].tolist(), lang_from=df_lang['lang'].tolist()[0], lang_to=lang)

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
    
    # Save results
    fname = os.path.join(storage_path, "Translated_data.csv")
    results.to_csv(fname, index = False)
    
    print("\n\n")
    


def generate_random_strings_wrapper(storage_path, num_strings=1000):
    """
    Wrapper function to generate random strings with the label -1 and save them.

    Parameters:
    storage_path (str): Where should the results be stored?
    num_strings (int, optional): Number of random strings to generate. Defaults to 1000.
    string_length (int, optional): Length of each random string. Defaults to 10.

    Returns:
    None
    """
    random_strings_df = generate_random_strings(num_strings)
    
    fname = os.path.join(storage_path, "Random_strings.csv")
    random_strings_df.to_csv(fname, index = False)

    print(f"Generated {num_strings} random strings with label -1.")
