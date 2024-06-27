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
        pred_aug = self.predictor.predict(aug_text, lang = self.lang, what = "bin")[0].tolist()
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

        test2 = all([x==y for x, y in zip(self.pred0[i], labels)]) # FIXME check similiarity based on membership (not order dependent)

        return test1 and test2

    def get_adv_string(self, text, lang, labels, i, verbose_extra = False, n_max = 100, alt_prob = 1, n_trans=3, double_translate = True):
        """
        Runs adversarial attack on a piece of text.

        Parameters:
        text (str): The text to attack.
        lang (str): The language of the text.
        labels (list): The expected labels.
        verbose_extra (bool): If True, prints additional information.
        n_max (int): Maximum number of augmentation attempts.
        alt_prob (float): Probability of strings transformation in attacker.attack()
        n_trans (float): Number of transformations attacker.attack()
        double_translate (bool): If True, double translation is used as the primary augmentation.


        Returns:
        tuple: The adversarial text and the number of attempts.
        """
        self.lang = lang

        if lang == 'en':
            raise NotImplementedError("English not implemented yet. Would translate from English to English")

        # Validate augmentation
        test = self._validate_augmentation(text, labels, i)
        if not test:
            return None, -1

        # Predict HISCO codes from raw string
        pred0_bin = self.predictor.predict([text], lang = lang, what = "bin")[0].tolist()
        aug_text = text

        for i in range(n_max):
            if double_translate:
                aug_text1 = self.translator.double_translate(aug_text, self.lang, lang_to="en")
                if aug_text1 == aug_text: # Then it is stuck in a loop
                    aug_text1 = self.attacker.attack(aug_text, alt_prob = alt_prob, n_trans=n_trans)
                aug_text = aug_text1
            else:
                aug_text = self.attacker.attack(aug_text, alt_prob = alt_prob, n_trans=n_trans)

            if self._goalfunction(pred0_bin, aug_text):
                if verbose_extra:
                    print(f"Succeful attack after {i+1} augments:\n{text} --> {aug_text[0]}")

                    pred = self.predictor.predict([text], lang = lang, get_dict = True)
                    print("")
                    print("Initial label prediction:")
                    print(pred)

                    print("")
                    print("Attacked label prediction:")
                    pred = self.predictor.predict(aug_text, lang = lang, get_dict = True)
                    print(pred)
                    print("")

                break

        return aug_text[0], (i+1)


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


def load_training_data(folder = "../Data/Training_data", toyload=False, verbose = True, sample_size = None):
    """
    Loads and combines training data from CSV files in the specified folder.

    Parameters:
    folder (str): The folder containing the CSV files.
    toyload (bool): If True, loads only a small subset of data for testing.
    verbose (bool): If True, prints progress information.
    sample_size (float): Number of observations to read. Defaults to reading all

    Returns:
    pd.DataFrame: The combined dataframe of training data.
    """
    fnames = os.listdir(folder)

    if sample_size:
        share_of_sample = sample_size // len(fnames)

    # Initialize an empty dataframe to store the data
    combined_df = pd.DataFrame()
    # Loop through the file list and read each CSV file into the combined dataframe
    for file in fnames:
        if file.endswith(".csv"):  # Make sure the file is a CSV file
            file_path = os.path.join(folder, file)

            if toyload:
                df = pd.read_csv(file_path, nrows = 2)#10000)
            else:
                df = pd.read_csv(file_path)

            df = df.drop(columns=['RowID'])
            df = df.drop_duplicates()
            df["source"] = file

            # Handling 'share_of_sample' larger than samples in n
            actual_n = df.shape[0] if share_of_sample > df.shape[0] else share_of_sample

            df = df.sample(n = actual_n, replace=False)

            combined_df = pd.concat([combined_df, df])

            n_df = df.shape[0]

            if verbose:
                print(f"\nRead {file} (N = {n_df})")

    df = combined_df

    # Make sure that all occ1 are strings
    df["occ1"] = [str(x) for x in df["occ1"].tolist()]

    return df


def generate_adversarial_wrapper(toyload=False, double_translate = True, verbose_extra=False, sample_size = 1000, n_max = 10):
    """
    Main function to generate adversarial examples.

    Parameters:
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

    df = load_training_data(toyload=toyload, sample_size=sample_size)
    df = df.drop_duplicates()

    n_df = df.shape[0]
    print(f"\nRead everything (N = {n_df})")

    # Init translator
    translator = Translator()

    # Init attacker class
    attacker = AttackerClass(df)

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

        print(f"{eta(i=i, start_time=start_time, cap_n=cap_n)} --> Stopped after {n_augs} of {n_max}")
        i += 1

    # Reset the index of the results DataFrame if necessary
    results.reset_index(drop=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    results = pd.concat([df, results], ignore_index=True, axis=1)

    results.to_csv(f"../Data/Adversarial_data/Adv_data_double_translate{double_translate}.csv")


def translated_strings_wrapper(toyload=False, verbose=True, sample_size = 1000):
    """
    Translates occupation strings in a dataset to multiple languages using a predefined translator.

    Parameters:
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

    df = load_training_data(toyload=toyload, sample_size = sample_size)
    df = df.drop_duplicates()
    df = df[df['lang'] != 'unk']

    n_df = df.shape[0]
    print(f"\nRead everything (N = {n_df})")

    results = pd.DataFrame()
    for lang in translator.lang_mapping:
        if lang == 'unk':
            continue
        df_lang = df[df['lang'] != lang]  # Only translate those that are not already in the target language
        translated_occ = translator.translate(df_lang['occ1'].tolist(), lang_from=df_lang['lang'].tolist()[0], lang_to=lang)

        if verbose:
            print(f"--> Finished translating to {lang}")

        res_i = pd.DataFrame(translated_occ) # Convert to DataFrame
        results = pd.concat([results, res_i], ignore_index=True)

    results.reset_index(drop=True, inplace=True)

    # Save results
    df.to_csv("../Data/Adversarial_data/Translated_data.csv")


def generate_random_strings_wrapper(num_strings=1000):
    """
    Wrapper function to generate random strings with the label -1 and save them.

    Parameters:
    num_strings (int, optional): Number of random strings to generate. Defaults to 1000.
    string_length (int, optional): Length of each random string. Defaults to 10.

    Returns:
    None
    """
    random_strings_df = generate_random_strings(num_strings)
    random_strings_df.to_csv("../Data/Adversarial_data/Random_strings.csv", index=False)

    print(f"Generated {num_strings} random strings with label -1.")


def main():
    # Using attacker.attack to generates adv. examples:
    generate_adversarial_wrapper(double_translate=False, sample_size = 100000, n_max = 20)

    # Using double translation to generates adv. examples (takes longer):
    generate_adversarial_wrapper(double_translate=True, sample_size = 10000, n_max = 20)

    # Generating strings in every language (takes longer)
    translated_strings_wrapper(sample_size = 100000)

    generate_random_strings_wrapper(num_strings=1000000)


if __name__ == '__main__':
    main()
