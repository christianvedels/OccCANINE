# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 13:33:15 2023

@author: chris
"""

import random
import string

# Build the Sentiment Classifier class
class AttackerClass:
    # Constructor class
    def __init__(self, df):
        # super(AttackerClass, self).__init__()
        all_text = ' '.join(str(item) for item in df['occ1'].tolist())
        self.word_list = all_text.split()

    # attack
    def attack(self, x_string, alt_prob = 0.1, insert_words = True):
        # breakpoint()
        x_string = [x_string]
        x_string_copy = x_string.copy()

        if(alt_prob == 0): # Then don't waste time
            return(x_string_copy)

        # Alter chars
        for i in range(len(x_string_copy)):
            # alt_prob probability that nothing will happen to the string
            if random.random() < alt_prob:
                continue

            # breakpoint()
            string_i = x_string_copy[i]

            num_letters = len(string_i)
            num_replacements = int(num_letters * alt_prob)

            indices_to_replace = random.sample(range(num_letters), num_replacements)

            # Convert string to list of characters
            chars = list(string_i)

            for j in indices_to_replace:
                chars[j] =  random.choice(string.ascii_lowercase) # replace with a random letter

            string_i = ''.join(chars)

            x_string_copy[i] = string_i

        if insert_words:
            for i in range(len(x_string_copy)):
                if random.random() < alt_prob: # Only make this affect alt_prob of cases
                    # Word list
                    occ_as_word_list = x_string_copy[i].split()

                    # Random word
                    random_word = random.choice(self.word_list)

                    # choose a random index to insert the word
                    insert_index = random.randint(0, len(occ_as_word_list))

                    # insert the word into the list
                    occ_as_word_list.insert(insert_index, random_word)

                    x_string_copy[i] = " ".join(occ_as_word_list)

        return x_string_copy
