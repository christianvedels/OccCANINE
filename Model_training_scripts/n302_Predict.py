# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 16:04:32 2023

@author: christian-vs
"""
# %%
import os
script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)

# %% Libraries and modules
from n001_Model_assets import *
from n101_Trainer import *
from n102_DataLoader import *
from n100_Attacker import AttackerClass
import pandas as pd
from transformers import AutoTokenizer

#%% Hyperparameters

# Which training data is used for the model
# MODEL_DOMAIN = "HSN_DATABASE"
# MODEL_DOMAIN = "DK_CENSUS"
# MODEL_DOMAIN = "EN_MARR_CERT"
MODEL_DOMAIN = "Multilingual"

# Parameters
SAMPLE_SIZE = 6 # 10 to the power of this is used for training
EPOCHS = 500
BATCH_SIZE = 2**5
LEARNING_RATE = 2*10**-5
UPSAMPLE_MINIMUM = 0
ALT_PROB = 0.1
INSERT_WORDS = True
DROPOUT_RATE = 0 # Dropout rate in final layer
MAX_LEN = 64 # Number of tokens to use

if MODEL_DOMAIN == "Multilingual":
    MODEL_NAME = f'XML_RoBERTa_{MODEL_DOMAIN}_sample_size_{SAMPLE_SIZE}_lr_{LEARNING_RATE}_batch_size_{BATCH_SIZE}' 
else: 
    MODEL_NAME = f'BERT_{MODEL_DOMAIN}_sample_size_{SAMPLE_SIZE}_lr_{LEARNING_RATE}_batch_size_{BATCH_SIZE}' 

BATCH_SIZE = 2 # Actual batch size used other than the name

key0 = pd.read_csv("../Data/Key.csv")

#%% Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

# %% Load data + tokenizer
data = Load_data(
    model_domain = MODEL_DOMAIN,
    downsample_top1 = True,
    upsample_below = UPSAMPLE_MINIMUM,
    sample_size = SAMPLE_SIZE,
    max_len = MAX_LEN,
    alt_prob = ALT_PROB,
    insert_words = INSERT_WORDS,
    batch_size = BATCH_SIZE,
    verbose = False
    )

# # # Sanity check
# for d in data['data_loader_train_attack']: 
#     print(d['occ1'][0][0])

# %% Load tokenizer
tokenizer_save_path = '../Trained_models/' + MODEL_NAME + '_tokenizer'
tokenizer = AutoTokenizer.from_pretrained(tokenizer_save_path)

# Temp code 
# tokenizer = data['tokenizer']
test = "This is a sentence"

tokenizer(test)

# %% Load best model instance
# Define the path to the saved binary file
model_path = '../Trained_models/'+MODEL_NAME+'.bin'

# Load the model
loaded_state = torch.load(model_path)

if MODEL_DOMAIN == "Multilingual":
    model_best = XMLRoBERTaOccupationClassifier(
        n_classes = data['N_CLASSES'], 
        model_domain = MODEL_DOMAIN, 
        tokenizer = tokenizer, 
        dropout_rate = DROPOUT_RATE
        )
else:
    model_best = BERTOccupationClassifier(
        n_classes = data['N_CLASSES'], 
        model_domain = MODEL_DOMAIN, 
        tokenizer = tokenizer, 
        dropout_rate = DROPOUT_RATE
    )
    
model_best.load_state_dict(loaded_state)

model_best.to(device)
# Set model to evaluation mode
model_best.eval()



# %% Get model prediction
def get_predictions(inputs):
    # Get model's prediction for the attacked example
    
    input_ids = inputs['input_ids']
    attention_mask = inputs["attention_mask"]
    
    with torch.no_grad():
        logits = model_best(input_ids, attention_mask)
    predicted_probs = torch.sigmoid(logits)
    threshold = 0.5  # You can adjust this threshold based on your use case
    predicted_labels = [key[i] for i, prob in enumerate(predicted_probs[0]) if prob > threshold]
    return predicted_labels

    
# %% Run it



examples = [Concat_string(x, "unk") for x in examples]

inputs = examples[0]

inputs = tokenizer(inputs, return_tensors='pt', padding=True, truncation=True)

input_ids = inputs['input_ids']
attention_mask = inputs["attention_mask"]

key = data['key']
 
with torch.no_grad():
    logits = model_best(input_ids, attention_mask)
predicted_probs = torch.sigmoid(logits)
threshold = 0.5  # You can adjust this threshold based on your use case
predicted_labels = [key[i] for i, prob in enumerate(predicted_probs[0]) if prob > threshold]
print(predicted_labels)

attacker = AttackerClass(df)

Attention_Viz(Concat_string('no longer active but used to be a blacksmith', "en"))
 
Attention_Viz2(Concat_string('no longer active but used to be a blacksmith', "en"))
Attention_Viz2(examples[4])
