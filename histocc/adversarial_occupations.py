# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 13:56:27 2024

@author: christian-vs
"""

from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import pandas as pd
import torch

class Translator():
    
    # Paper: https://arxiv.org/pdf/2010.11125
    
    def __init__(self, device=None):
        self.model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
        self.tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

        # translate Danish to English
        self.tokenizer.src_lang = "da"
        self.lang_to = self.tokenizer.get_lang_id("en")
        
        # Detect device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)
        print(f"Using device: {self.device}")
        self.model.to(self.device)
        
    def translate(self, text, lang_from, lang_to):
        # Handle 'unk' language
        if lang_from=='unk' or lang_to=='unk':
            NotImplementedError("A way of handling unknown languages is not implemented yet")
        
        # Test types
        if isinstance(text, str):
            text = [text]
        if not isinstance(text, list):
            raise TypeError("Should be 'list' or 'str'")
        
        # Set language from
        self.tokenizer.src_lang = lang_from        
        
        # Set language to
        target_lang = self.tokenizer.get_lang_id(lang_to)
        
        encoded = [self.tokenizer(x, return_tensors="pt") for x in text]
        encoded = [{k: v.to(self.device) for k, v in e.items()} for e in encoded]
        generated_tokens = [self.model.generate(**x, forced_bos_token_id=target_lang) for x in encoded]
        res = [self.tokenizer.batch_decode(x, skip_special_tokens=True) for x in generated_tokens]
        
        return(res)
    
    def double_translate(self, text, lang_from, lang_to):
        # Tranlastes back and forth
        step1 = self.translate(text, lang_from = lang_from, lang_to = lang_to)
        step2 = self.translate(step1, lang_from = lang_to, lang_to = lang_from)
        
        return step2
    
    def _convert_lang_abr(self, lang):
        # Converts a language abbreviation from OccCANINE to the abbreviation used in facebook/m2m100_418M
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
            'pt': 'pt',  # Portugues 
            'se': 'sv'  # Swedish 
        }
        return lang_mapping.get(lang, lang)  # Default to returning the same abbreviation if not found
   
if __name__ == '__main__':
    x = Translator()
    df = pd.read_csv("../Data/Training_data/DK_orsted_train.csv", nrows=100)
    res = x.translate(df.occ1.tolist(), lang_from = "da", lang_to = "en") 
