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

key0 = pd.read_csv("../Data/Key.csv")

#%% Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"

# %% Load data + tokenizer

key, df, df_bin = Load_val(
    model_domain = MODEL_DOMAIN,
    sample_size = 5 # SAMPLE_SIZE
    )

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
        n_classes = len(key), 
        model_domain = MODEL_DOMAIN, 
        tokenizer = tokenizer, 
        dropout_rate = DROPOUT_RATE
        )
else:
    model_best = BERTOccupationClassifier(
        n_classes = len(key), 
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
    predicted_labels = []
    BATCH_SIZE0 = BATCH_SIZE*16 # Prediction can handle larger batch size
    
    total_batches = (len(inputs) + BATCH_SIZE0 - 1) // BATCH_SIZE0  # Calculate the total number of batches

    for batch_num, i in enumerate(range(0, len(inputs), BATCH_SIZE0), 1):
        batch_inputs = inputs[i:i + BATCH_SIZE0]

        # Tokenize the batch of inputs
        batch_tokenized = tokenizer(batch_inputs, padding=True, truncation=True, return_tensors='pt')

        batch_input_ids = batch_tokenized['input_ids'].to(device)
        batch_attention_mask = batch_tokenized['attention_mask'].to(device)

        with torch.no_grad():
            batch_logits = model_best(batch_input_ids, batch_attention_mask)
            
        batch_predicted_probs = torch.sigmoid(batch_logits).cpu().numpy()
        threshold = 0.5  # You can adjust this threshold based on your use case

        for probs in batch_predicted_probs:
            labels = [key[i] for i, prob in enumerate(probs) if prob > threshold]
            predicted_labels.append(labels)

        if batch_num % 1 == 0:
            print(f"Processed batch {batch_num} out of {total_batches} batches")

    return predicted_labels

    
# %% Run it
x0 = get_predictions(
    inputs=df['concat_string0'].tolist()
    )

x1 = get_predictions(
    inputs=df['concat_string1'].tolist()
    )


# %% Convert to df
def convert_to_df(code_list, ID = 0):
    for i, code_element in enumerate(code_list):
        code_list[i] = code_element + [''] * (5 - len(code_element))

    # Create a DataFrame
    df = pd.DataFrame(
        code_list, 
        columns=[f'hisco_pred1_{ID}', f'hisco_pred2_{ID}', f'hisco_pred3_{ID}', f'hisco_pred4_{ID}', f'hisco_pred5_{ID}']
        )
    
    return df

df0 = convert_to_df(x0, 0)
df1 = convert_to_df(x1, 1)
# Reset the index of the original DataFrames
df0 = df0.reset_index(drop=True)
df1 = df1.reset_index(drop=True)

# Concatenate the DataFrames along the columns
joined_df = pd.concat([df0, df1], axis=1)

# If df_data also has a multi-column index, reset its index as well
df_data = data_val[0].reset_index(drop=True)

# Concatenate df_data and joined_df along the columns
resulting_df = pd.concat([df_data, joined_df], axis=1)

# %% Save
fname = f'../Data/Predictions/XML_RoBERTa_{MODEL_DOMAIN}_sample_size_{SAMPLE_SIZE}_lr_{LEARNING_RATE}_batch_size_{BATCH_SIZE}.csv' 

resulting_df.to_csv(fname, index = False)