# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 13:33:15 2023

@author: chris
"""


import random
import string

import pandas as pd


class AttackerClass:
    """
    A class used to perform various text attack transformations on a DataFrame.
    """
    def __init__(self, df):
        """
        Initializes the AttackerClass with a DataFrame.

        Parameters:
        df (pd.DataFrame): A DataFrame containing text data in a column named 'occ1'.
        """
        all_text = ' '.join(str(item) for item in df['occ1'].tolist())
        self.word_list = all_text.split()

    # Helper functions for transformations
    def random_character_deletion(self, sentence):
        """
        Randomly deletes a character from the sentence.

        Parameters:
        sentence (str): The input sentence.

        Returns:
        str: The sentence with one character randomly deleted.
        """
        if not sentence:
            return sentence
        index = random.randint(0, len(sentence) - 1)
        return sentence[:index] + sentence[index + 1:]

    def qwerty_substitution(self, sentence):
        """
        Substitutes a character in the sentence with a neighboring character on a QWERTY keyboard.

        Parameters:
        sentence (str): The input sentence.

        Returns:
        str: The sentence with one character substituted by a neighboring QWERTY character.
        """
        qwerty = {
            'a': 'qwsz', 'b': 'vghn', 'c': 'xdfv', 'd': 'serfcx', 'e': 'wsdr',
            'f': 'rtgvcd', 'g': 'tyhbvf', 'h': 'yujnbg', 'i': 'ujklo', 'j': 'uikmnh',
            'k': 'ijolm', 'l': 'kop', 'm': 'njkl', 'n': 'bhjm', 'o': 'iklp',
            'p': 'ol', 'q': 'wa', 'r': 'edft', 's': 'wedxz', 't': 'rfgy',
            'u': 'yhji', 'v': 'cfgb', 'w': 'qase', 'x': 'zsdc', 'y': 'tghu',
            'z': 'asx'
        }
        if not sentence:
            return sentence

        index = random.randint(0, len(sentence) - 1)
        char = sentence[index]

        if char in qwerty:
            substitute = random.choice(qwerty[char])
            return sentence[:index] + substitute + sentence[index + 1:]

        return sentence

    def random_character_insertion(self, sentence):
        """
        Inserts a random character into the sentence.

        Parameters:
        sentence (str): The input sentence.

        Returns:
        str: The sentence with one random character inserted.
        """
        if not sentence:
            return sentence

        index = random.randint(0, len(sentence))
        char = random.choice(string.ascii_lowercase)

        return sentence[:index] + char + sentence[index:]

    def random_character_substitution(self, sentence):
        """
        Substitutes a random character in the sentence with another random character.

        Parameters:
        sentence (str): The input sentence.

        Returns:
        str: The sentence with one character substituted by another random character.
        """
        if not sentence:
            return sentence

        index = random.randint(0, len(sentence) - 1)
        char = random.choice(string.ascii_lowercase)

        return sentence[:index] + char + sentence[index + 1:]

    def neighboring_character_swap(self, sentence):
        """
        Swaps two neighboring characters in the sentence.

        Parameters:
        sentence (str): The input sentence.

        Returns:
        str: The sentence with two neighboring characters swapped.
        """
        if len(sentence) < 2:
            return sentence

        index = random.randint(0, len(sentence) - 2)

        return sentence[:index] + sentence[index + 1] + sentence[index] + sentence[index + 2:]

    def word_swap(self, sentence):
        """
        Swaps two neighboring words in the sentence.

        Parameters:
        sentence (str): The input sentence.

        Returns:
        str: The sentence with two neighboring words swapped.
        """
        words = sentence.split()
        if len(words) < 2:
            return sentence

        index = random.randint(0, len(words) - 2)
        words[index], words[index + 1] = words[index + 1], words[index]

        return ' '.join(words)

    def insert_random_word(self, sentence):
        """
        Inserts a random word from the word list into the sentence.

        Parameters:
        sentence (str): The input sentence.

        Returns:
        str: The sentence with a random word inserted.
        """
        occ_as_word_list = sentence.split()
        random_word = random.choice(self.word_list)
        insert_index = random.randint(0, len(occ_as_word_list))
        occ_as_word_list.insert(insert_index, random_word)

        return " ".join(occ_as_word_list)

    def apply_transformations(self, sentence, n_trans):
        """
        Applies a random sequence of transformations to the sentence.

        Parameters:
        sentence (str): The input sentence.
        n_trans (int): The number of transformations to apply.

        Returns:
        str: The transformed sentence.
        """
        transformations = [
            self.random_character_deletion,
            self.qwerty_substitution,
            self.random_character_insertion,
            self.random_character_substitution,
            self.neighboring_character_swap,
            self.word_swap,
            self.insert_random_word
        ]

        for _ in range(n_trans):
            transformation = random.choice(transformations)
            sentence = transformation(sentence)

        return sentence

    # attack
    def attack(self, x_string, alt_prob=0.8, n_trans=3):
        """
        Performs an attack on the input strings by applying text transformations.

        Parameters:
        x_string (list or str): The input string or list of strings to attack.
        alt_prob (float): The probability of altering each string.
        n_trans (int): The number of transformations to apply.

        Returns:
        list: The list of attacked strings.
        """
        if not isinstance(x_string, list):
            x_string = [x_string]

        x_string_copy = x_string.copy()

        # Test if everything is strings
        for x in x_string_copy:
            if not isinstance(x, str):
                raise TypeError("One or more elements were not strings")

        if alt_prob == 0:  # Then don't waste time
            return x_string_copy

        # Augment strings
        for i in range(len(x_string_copy)):
            # alt_prob probability that something will happen to the string
            if random.random() > alt_prob:
                continue

            string_i = x_string_copy[i]

            string_i = self.apply_transformations(string_i, n_trans=n_trans)

            x_string_copy[i] = string_i

        return x_string_copy


def main():
    '''Test the AttackerClass
    '''
    df = pd.read_csv("../Data/Training_data/DK_orsted_train.csv", nrows=100)
    attacker = AttackerClass(df)

    test_strings = df['occ1'].tolist()

    attacked_strings = attacker.attack(test_strings, alt_prob=0.3, n_trans=10)

    print("\nAttacked strings:")

    for s in attacked_strings:
        print(s)


if __name__ == '__main__':
    main()
