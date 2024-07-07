# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 14:37:11 2024

@author: chris
"""

from .prediction_assets import OccCANINE
from .attacker import AttackerClass
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from IPython.display import display, HTML

class ClassActivation():
    
    def __init__(self):
        self.model = OccCANINE()        
        self.letters = self._letters()
        
        keys = pd.read_csv("Data/Key.csv")
        keys['occ1'] = [str(x) for x in keys.en_hisco_text.tolist()]
        all_text = ' '.join(item for item in keys['occ1'].unique())
        self.word_list = list(set(all_text.split()))
        self.keys = keys
        
        self.attacker = AttackerClass(alt_prob = 1, n_trans=3, df = keys)
        
    def _letters(self):
        x = string.ascii_lowercase+' '
        x = [*x]
        # x.append('')
        
        return x
            
    def _add_char_to_string(self, x):
        res = []
        for char in self.letters:
            res.append(x+char)
        
        return res  
    
    def _run_occ_canine(self, x, lang = "en"):
        # Strip white_space
        # x = [y.strip() for y in x]
        res = self.model.predict(x, lang=lang, what="probs")
        return res
        
    def _alter_string_to_increase_class_prob(self, x, class_index, verbose, lang = "en"):
        strings_to_test = [x]
        for i in range(len(x)):
            for k in self.letters:
                x = x[:i] + k + x[i + 1:]
                strings_to_test.append(x)
        
        preds = self._run_occ_canine(strings_to_test, lang = lang)
        
        max_prob = 0
        for i in range(len(strings_to_test)):
            class_prob_for_desc_i = preds[i][class_index]
            if class_prob_for_desc_i > max_prob:
                res = strings_to_test[i]
                max_prob = class_prob_for_desc_i
        
        if verbose:
            print(f"{max_prob:.3f}: {res}")
        
        return res, max_prob
    
    def _alter_string_attack(self, x, class_index, verbose, n_strings = 8192, lang = "en"):
        strings_to_test = [x]
        for i in range(n_strings):
            strings_to_test.append(self.attacker.attack(x))
        
        preds = self._run_occ_canine(strings_to_test, lang = lang)
        
        max_prob = 0
        for i in range(len(strings_to_test)):
            class_prob_for_desc_i = preds[i][class_index]
            if class_prob_for_desc_i > max_prob:
                res = strings_to_test[i]
                max_prob = class_prob_for_desc_i
        
        if verbose:
            print(f"{max_prob:.3f}: {res} (attack)")
        
        return res, max_prob
    
    def max_class_addition(self, class_index, lang = "en", verbose = True):
       desc = ""
       for i in range(60):
           desc = self._add_char_to_string(desc)
           preds = self._run_occ_canine(desc, lang = lang)
           max_prob = 0
           for k in range(len(preds)):
               class_prob_for_desc_k = preds[k][class_index]
               
               if class_prob_for_desc_k > max_prob:
                   new_desc = desc[k]
                   max_prob = class_prob_for_desc_k
               
           # Choose desc which maxes class
           if verbose:
               print(f"{max_prob:.3f}: {new_desc}")
           desc = new_desc
       
       return desc
   
    def _add_word(self, x, class_index, verbose, n_strings = 1024, lang = "en"):
        
        desc = [x]
        for w in self.word_list:
            desc.append(x + ' ' + w)
        
        preds = self._run_occ_canine(desc, lang = lang)
        max_prob = 0
        for i in range(len(desc)):
            class_prob_for_desc_i = preds[i][class_index]
            if class_prob_for_desc_i > max_prob:
                res = desc[i]
                max_prob = class_prob_for_desc_i
                
        if verbose:
            print(f"{max_prob:.3f}: {res}")
        
        return res, max_prob
       
        
    def max_class_change(self, class_index, lang = "en", verbose = True):
        # description = self.max_class_addition(class_index = class_index, verbose=verbose)
        description = ""
        
        print(description)
        
        for i in range(20):
            desc_new, prob = self._add_word(description, class_index, verbose)
            
            if desc_new == description:
                desc_new, prob = self._alter_string_to_increase_class_prob(description, class_index, verbose = verbose)
                if desc_new == description:
                    desc_new, prob = self._alter_string_attack(description, class_index, verbose = verbose)
            
            if prob > 0.999:
                break
            
            description = desc_new
            
        return description, prob
    
class StringGradient:
    """
    A class to compute and visualize gradients of n-grams in text with respect to a target class using a model.

    Attributes:
        model (OccCANINE): An instance of the OccCANINE model.
        letters (list): A list of all lowercase letters and a space character.
        max_n (int): The maximum length of n-grams to consider.
    """

    def __init__(self, n=1, max_n=2, verbose=False, verbose_extra = False):
        """
        Initializes the StringGradient class with a specified n-gram length and maximum n-gram length.

        Args:
            n (int): The length of n-grams to consider.
            verbose (bool): Should updates be printed?
            verbose_extra (bool): Should extra updates be printed (including OccCANINE run-info)?
        """
        self.model = OccCANINE(verbose=verbose_extra)
        self.letters = self._letters()
        self.n = n
        self.verbose = verbose
        self.ngrams = None
        self.grads = None

    def _letters(self):
        """
        Generates a list of all lowercase letters and a space character.

        Returns:
            list: A list of all lowercase letters and a space character.
        """
        x = string.ascii_lowercase + ' '
        x = [*x]
        return x
    
    def _change_ngram(self, x, where, what):
        """
        Changes an n-gram at a specific location in a string.

        Args:
            x (str): The original string.
            where (int): The index at which to change the n-gram.
            what (str): The new n-gram to insert.

        Returns:
            str: The modified string with the new n-gram.
        """
        x = x[:where] + what + x[where + len(what):]
        return x
    
    def _manual_grad_computation(self, x, loc, target_class, n, lang="en"):
        """
        Computes the gradient for an n-gram at a specific location in the input string.

        Args:
            x (str): The input string.
            loc (int): The location of the n-gram in the string.
            target_class (int): The target class index.
            n (int): The length of the n-gram.
            lang (str): The language of the input string.

        Returns:
            float: The computed gradient for the n-gram.
        """
        prob = self.model.predict([x], lang=lang, what="probs")[0][target_class]
        
        alternative_x = []
        for letters in self._generate_ngrams(self.letters, n):
            x_i = self._change_ngram(x, loc, letters)
            alternative_x.append(x_i)
            
        probs_alt = self.model.predict(alternative_x, lang=lang, what="probs")
        probs_alt = [prob[target_class] for prob in probs_alt]
        return prob - np.mean(probs_alt)  # How much larger is prob because of this n-gram?
    
    def _generate_ngrams(self, letters, n):
        """
        Generates all possible n-grams of a given length from the list of letters.

        Args:
            letters (list): The list of letters.
            n (int): The length of n-grams to generate.

        Returns:
            list: A list of all possible n-grams of length n.
        """
        return [''.join(ngram) for ngram in product(letters, repeat=n)]
    
    def compute_gradients(self, input_text, target_class, lang="en"):
        """
        Computes gradients for all n-grams in the input text with respect to the target class.

        Args:
            input_text (str): The input text.
            target_class (int): The target class index.
            lang (str): The language of the input text.

        Returns:
            tuple: A tuple containing a list of n-grams and their corresponding gradients.
        """
        grads = []
        ngrams = []
        for loc in range(len(input_text) - self.n + 1):
            grad_loc = self._manual_grad_computation(input_text, loc, target_class, self.n, lang=lang)
            grads.append(grad_loc)
            ngrams.append(input_text[loc:loc + self.n])
            
            if self.verbose:
                string_loc = ''.join([input_text[i] for i in range(loc, loc + self.n)])
                print(f'--> Computed pseudo-gradient for {string_loc}')
            
        self.ngrams = ngrams
        self.grads = grads
        
        return ngrams, grads
        
    def visualize_gradients(self, input_text, target_class, lang="en"):
        """
        Visualizes the gradients of n-grams in the input text using a bar chart.

        Args:
            input_text (str): The input text.
            target_class (int): The target class index.
            lang (str): The language of the input text.
        """
        ngrams, gradients = self.compute_gradients(input_text, target_class, lang)
        
        fig, ax = plt.subplots()
        norm = plt.Normalize(vmin=min(gradients), vmax=max(gradients))
        sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
        sm.set_array([])

        plt.bar(range(len(ngrams)), gradients, color=[sm.to_rgba(grad) for grad in gradients])
        plt.xticks(range(len(ngrams)), ngrams, rotation=90)
        plt.colorbar(sm, ax=ax, orientation='vertical')
        plt.xlabel('N-gram')
        plt.ylabel('Gradient Value')
        plt.title(f'{self.n}-gram Gradients Visualization')
        plt.show()
        
    def visualize_text_gradients(self, input_text, target_class, lang="en"):
        """
        Visualizes the gradients of n-grams in the input text with color-coded text.

        Args:
            input_text (str): The input text.
            target_class (int): The target class index.
            lang (str): The language of the input text.
        """
        ngrams, gradients = self.compute_gradients(input_text, target_class, lang)
        ngram_gradients = {input_text[loc:loc + self.n]: grad for loc, grad in enumerate(gradients)}

        # Aggregate gradients for each character
        letter_gradients = {char: 0.0 for char in input_text}
        letter_counts = {char: 0 for char in input_text}

        for loc in range(len(input_text) - self.n + 1):
            ngram = input_text[loc:loc + self.n]
            grad = ngram_gradients[ngram]
            for i, char in enumerate(ngram):
                letter_gradients[char] += grad / self.n
                letter_counts[char] += 1

        for char in letter_gradients:
            if letter_counts[char] > 0:
                letter_gradients[char] /= letter_counts[char]

        min_grad, max_grad = min(letter_gradients.values()), max(letter_gradients.values())

        def get_color(value):
            norm_value = (value - min_grad) / (max_grad - min_grad) if max_grad != min_grad else 0.5
            r = int(255 * (1 - norm_value))
            g = int(255 * norm_value)
            b = 0
            return (r / 255, g / 255, b / 255)

        # Create a plot
        fig, ax = plt.subplots()
        ax.axis('off')
        for i, char in enumerate(input_text):
            color = get_color(letter_gradients[char])
            ax.text(i * 0.05, 0.5, char, fontsize=20, ha='center', va='center', bbox=dict(facecolor=color, edgecolor=color))

        plt.show()
    
    # TODO: Which chars can be removed which decreases prob the least?
    # TODO: Use next word generation transformer to generate text