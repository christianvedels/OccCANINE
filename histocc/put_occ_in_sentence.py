# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 21:23:35 2024

@author: chris
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class Make_occ_sentence():
    """
    A class used to generate sentences based on a specific word and a given prompt using a pre-trained language model.
    
    Attributes
    ----------
    tokenizer : transformers.PreTrainedTokenizer
        The tokenizer used to preprocess the text input.
    model : transformers.PreTrainedModel
        The pre-trained language model for text generation.
    device : torch.device
        The device (CPU or GPU) on which the model will be run.
    """
    
    def __init__(self, model_id="HuggingFaceH4/zephyr-7b-beta"):
        """
        Initializes the Make_occ_sentence class by loading the tokenizer and model,
        and setting the device (CPU or GPU) for model inference.
        
        Parameters
        ----------
        model_id : str, optional
            The model identifier for the pre-trained language model (default is "HuggingFaceH4/zephyr-7b-beta").
        """
        # Load the model
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Initialize model
        model = AutoModelForCausalLM.from_pretrained(model_id)

        # Ensure the model is in evaluation mode and use a GPU if available
        model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(self.device)
        
        self.model = model
        
    def produce_sentence(self, specific_word="farmer", prompt=None, temperature=1.0):
        """
        Generates sentences based on the specific word and given prompt.
        
        Parameters
        ----------
        specific_word : str, optional
            The word to be defined or included in the prompt (default is "farmer").
        prompt : str, optional
            The prompt used to guide sentence generation. If None, a default prompt is used (default is None).
        temperature : float, optional
            The temperature parameter controls the randomness of predictions. A lower value makes the model's output more deterministic, while a higher value increases randomness (default is 1.0).
        
        Returns
        -------
        None
            Prints the generated sentences.
        """
        if prompt is None:
            prompt = f"Please define '{specific_word}':"
        else:
            prompt = prompt.format(specific_word=specific_word)
        
        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate the output
        with torch.no_grad():
            output = self.model.generate(
                inputs["input_ids"], 
                max_length=100, 
                num_return_sequences=10, 
                num_beams=10,
                early_stopping=True,
                no_repeat_ngram_size=2,
                temperature=temperature,
                do_sample=True,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode and print the generated sentences
        generated_sentences = [self.tokenizer.decode(g, skip_special_tokens=True) for g in output]
        for i, sentence in enumerate(generated_sentences):
            print(f"Sentence {i+1}: {sentence}")

# Init class
sentence_engine = Make_occ_sentence()

# Use class
sentence_engine.produce_sentence(
    specific_word="finest tailor of the town",
    prompt="The occupation, {specific_word}, is defined as:",
    temperature=1
)

sentence_engine.produce_sentence(
    specific_word="finest tailor of the town",
    prompt="A sentence containing '{specific_word}':",
    temperature=1
)
