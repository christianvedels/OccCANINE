# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 13:56:27 2024

@author: christian-vs
"""

from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import pandas as pd

df = pd.read_csv("../Data/Training_data/DK_orsted_train.csv")

class Translator():
    
    def __init__(self):
        self.model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
        self.tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

        # translate Danish to English
        self.tokenizer.src_lang = "da"
        self.lang_to = self.tokenizer.get_lang_id("en")
        
    def translate(self, text, lang_from, lang_to):
        self.tokenizer.src_lang = lang_from        
        target_lang = self.tokenizer.get_lang_id(lang_to)
        encoded = self.tokenizer(text, return_tensors="pt")
        generated_tokens = self.model.generate(**encoded, forced_bos_token_id=target_lang)
        res = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        return(res)
    
x = Translator()   


x.translate("hej med dig", lang_from = "da", lang_to = "en") 

