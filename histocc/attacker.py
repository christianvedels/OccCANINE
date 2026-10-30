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

    qwerty = {
        'a': 'qwsz', 'b': 'vghn', 'c': 'xdfv', 'd': 'serfcx', 'e': 'wsdr',
        'f': 'rtgvcd', 'g': 'tyhbvf', 'h': 'yujnbg', 'i': 'ujklo', 'j': 'uikmnh',
        'k': 'ijolm', 'l': 'kop', 'm': 'njkl', 'n': 'bhjm', 'o': 'iklp',
        'p': 'ol', 'q': 'wa', 'r': 'edft', 's': 'wedxz', 't': 'rfgy',
        'u': 'yhji', 'v': 'cfgb', 'w': 'qase', 'x': 'zsdc', 'y': 'tghu',
        'z': 'asx',
    }

    def __init__(
            self,
            alt_prob: float = 0.8,
            n_trans: int = 3,
            df: pd.DataFrame | None = None,
            word_freq_table: pd.DataFrame | None = None,
    ):
        """
        Initializes the AttackerClass with a DataFrame.

        Parameters:
        df (pd.DataFrame): A DataFrame containing text data in a column named 'occ1'.
        word_freq_table (pd.DataFrame): A DataFrame with two columns, "word" and "freq",
        from which to draw random words. If this is applied, the argument df is ignored.
        """

        self.alt_prob = alt_prob
        self.n_trans = n_trans

        self.transformations = [
            self.random_character_deletion,
            self.qwerty_substitution,
            self.random_character_insertion,
            self.random_character_substitution,
            self.neighboring_character_swap,
            self.word_swap,
            self.add_leading_trailing_characters,
        ]

        if word_freq_table is not None:
            # Always use frequency table when available
            df = None

        if df is not None:
            all_text = ' '.join(item for item in df['occ1'].unique())
            self.word_list = list(set(all_text.split()))
            self.transformations.append(self.insert_random_word)

        self.word_freq_table = word_freq_table

        if self.word_freq_table is not None:
            self.transformations.append(self.insert_random_word)

    @staticmethod
    def random_character_deletion(sentence: str) -> str:
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

    def qwerty_substitution(self, sentence: str) -> str:
        """
        Substitutes a character in the sentence with a neighboring character on a QWERTY keyboard.

        Parameters:
        sentence (str): The input sentence.

        Returns:
        str: The sentence with one character substituted by a neighboring QWERTY character.
        """

        if not sentence:
            return sentence

        index = random.randint(0, len(sentence) - 1)
        char = sentence[index]

        if char in self.qwerty:
            substitute = random.choice(self.qwerty[char])
            return sentence[:index] + substitute + sentence[index + 1:]

        return sentence

    @staticmethod
    def random_character_insertion(sentence: str) -> str:
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

    @staticmethod
    def random_character_substitution(sentence: str) -> str:
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

    @staticmethod
    def neighboring_character_swap(sentence: str) -> str:
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

    @staticmethod
    def word_swap(sentence: str) -> str:
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

    def insert_random_word(self, sentence: str) -> str:
        """
        Inserts a random word from the word list into the sentence.

        Parameters:
        sentence (str): The input sentence.

        Returns:
        str: The sentence with a random word inserted.
        """
        occ_as_word_list = sentence.split()

        if self.word_freq_table is not None:
            random_word = self.word_freq_table.sample(n=1, weights='freq')['word'].iloc[0]
        else:
            random_word = random.choice(self.word_list)

        insert_index = random.randint(0, len(occ_as_word_list))
        occ_as_word_list.insert(insert_index, random_word)

        return " ".join(occ_as_word_list)

    @staticmethod
    def add_leading_trailing_characters(sentence: str, nchars = 20) -> str:
        """
        Adds 1 to 'nchars' leading and trailing random lowercase ASCII characters to the sentence.
        There is a 50 percent chance of leading characters being added and a 50 percent chance
        for trailing characters. If both fail (25 percent chance) then both leading and
        trailing characters will be added

        Parameters:
        sentence (str): The input sentence.
        nchars (int): Number of chars to add (defaults to 20)

        Returns:
        str: The sentence with added leading and trailing random characters.
        """
        characters = string.ascii_lowercase + ' ' # All lowercase + space

        if random.random() > 0.5:
            leading_chars = ''.join(random.choices(characters, k=random.randint(1, nchars)))
        else:
            leading_chars = ''

        if random.random() > 0.5:
            trailing_chars = ''.join(random.choices(characters, k=random.randint(1, nchars)))
        else:
            trailing_chars = ''

        if leading_chars == '' and trailing_chars == '':
            leading_chars = ''.join(random.choices(characters, k=random.randint(1, nchars)))
            trailing_chars = ''.join(random.choices(characters, k=random.randint(1, nchars)))

        return leading_chars + ' ' + sentence + ' ' + trailing_chars

    def apply_transformations(self, sentence: str, n_trans: int) -> str:
        """
        Applies a random sequence of transformations to the sentence.

        Parameters:
        sentence (str): The input sentence.
        n_trans (int): The number of transformations to apply.

        Returns:
        str: The transformed sentence.
        """

        for _ in range(n_trans):
            transformation = random.choice(self.transformations)
            sentence = transformation(sentence)

        return sentence

    def attack(self, input_string: str) -> str:
        if random.random() > self.alt_prob:
            return input_string

        return self.apply_transformations(input_string, n_trans=self.n_trans)

    def attack_multiple(self, x_string: str | list[str], alt_prob: float = 0.8, n_trans: int = 3) -> list[str]:
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
    df = pd.read_csv(r"Data/TOYDATA.csv", nrows=10)
    attacker = AttackerClass(df=df)

    test_strings = df['occ1'].tolist()

    attacked_strings = attacker.attack_multiple(test_strings, alt_prob=1, n_trans=1)

    print("\nAttacked strings:")

    for s in attacked_strings:
        print(s)


if __name__ == '__main__':
    main()
