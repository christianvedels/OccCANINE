# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 09:58:40 2023

https://huggingface.co/papluca/xlm-roberta-base-language-detection
https://colab.research.google.com/drive/15LJTckS6gU3RQOmjLqxVNBmbsBdnUEvl?usp=sharing#scrollTo=V_gbHRmNHEWU

https://github.com/abhimishra91/transformers-tutorials/blob/master/transformers_multi_label_classification.ipynb

@author: christian-vs
"""

# %% Parameters
n_epochs = 10 # The networks trains a maximum of these epochs

sample_size = 3
batch_size = 16

# %% Import packages
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset 
import os
import random as r
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    DataCollatorWithPadding, 
    pipeline,
    Trainer,
    TrainingArguments
)
import transformers
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# %% Check cuda availability
is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

device = torch.device("cpu")    

#%% Load data
if os.environ['COMPUTERNAME'] == 'SAM-126260':
    os.chdir('D:/Dropbox/PhD/HISCO clean')
else:
    os.chdir('C:/Users/chris/Dropbox/PhD/HISCO clean')
print(os.getcwd())

fname = "Data/Training_data/DK_census_train.csv"
df = pd.read_csv(fname, encoding = "UTF-8")

key = pd.read_csv("Data/Key.csv") # Load key and convert to dictionary
key = key[1:]
key = zip(key.code, key.hisco)
key = list(key)
key = dict(key)
label2id = key
id2label = {v: k for k, v in label2id.items()}
 

# Subset to smaller
if(10**sample_size < df.shape[0]):
    r.seed(20)
    df = df.sample(10**sample_size)

# Remove no occ
# df = df[df.code1==2]

# Split into internal test/train/val
train_ratio = 0.75
validation_ratio = 0.15
test_ratio = 0.10

# train is now 75% of the entire data set
df_train, df_test = train_test_split(df, test_size=1 - train_ratio)

# test is now 10% of the initial data set
# validation is now 15% of the initial data set
df_val, df_test = train_test_split(df_test, test_size=test_ratio/(test_ratio + validation_ratio)) 
print(f"Train / valid / test samples: {len(df_train)} / {len(df_val)} / {len(df_test)}")

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

# %% Labels to list
# Each person can have up to 5 occupations. This part converts this into binary representation in a NxK matrix
def labels_to_bin(df, max_value):
    df_codes = df[["code1", "code2", "code3", "code4", "code5"]]

    # Binarize
    labels_list = df_codes.values.tolist()

    # == Build outcome matrix ==
    # Convert the given list to a NumPy array
    labels_array = np.array(labels_list)

    # Find the maximum integer value in the given array
    max_value = max_value

    # Construct the NxK matrix
    N = len(labels_list)
    K = len(label2id)
    labels = np.zeros((N, K), dtype=int)

    for i, row in enumerate(labels_list):
        for value in row:
            if not np.isnan(value):
                labels[i, int(value)] = 1


    # The columns 0-2 contains all those labelled as having 'no occupation'.
    # But since any individuals is encoded with up 5 occupations, and any one o
    # these can be 'no occupation' it occurs erroneously with high frequency.
    # Fix: Check if any other occupation is positve.
    # Iterate through each row of the array
    for row in labels:
        # Check if the value in the first column is '1'
        if row[2] == 1 | row[1] == 1 | row[0] == 1:
            # Check if there are any positive values in the remaining row (excluding the first column)
            if np.any(row[3:]>0):
                # Set the first column to '0' if positive values are found
                row[0] = 0

    labels = labels.astype(float)
    
    # Convert each row of the array to a 1D list
    # labels = [row.tolist() for row in labels]
    
    return(labels)


# Run 
max_val = len(key)
labels_bin = labels_to_bin(df, max_value = max_val)
labels_train = labels_to_bin(df_train, max_value = max_val)
labels_val = labels_to_bin(df_val, max_value = max_val)
labels_test = labels_to_bin(df_test, max_value = max_val)

#%% Tokenizer
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')

# %% Key vars
MAX_LEN = 200
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 1e-05

# %% Defining CustomDataset 
# (https://github.com/abhimishra91/transformers-tutorials/blob/master/transformers_multi_label_classification.ipynb)

class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len, Labels):
        self.tokenizer = tokenizer
        self._data = dataframe
        self.occ1 = dataframe.occ1
        self.targets = Labels
        self.max_len = max_len

    def __len__(self):
        return len(self.occ1)

    def __getitem__(self, index):
        occ1 = str(self.occ1[index])
        occ1 = " ".join(occ1.split())

        inputs = self.tokenizer.encode_plus(
            occ1,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        # breakpoint()

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }
    
    
# %% Creating the dataset and dataloader for the neural network

training_set = CustomDataset(df_train, tokenizer, MAX_LEN, labels_train)
testing_set = CustomDataset(df_val, tokenizer, MAX_LEN, labels_val)

# %% Train params
train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)


# %% Model class

class xmlRoBERTaClass(torch.nn.Module):
    def __init__(self):
        super(xmlRoBERTaClass, self).__init__()
        self.l1 = transformers.XLMRobertaModel.from_pretrained('xlm-roberta-base')
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, len(key))
    
    def forward(self, ids, mask):
        # breakpoint()
        
        _, output_1= self.l1(ids, attention_mask = mask, return_dict=False)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output

model = xmlRoBERTaClass()
model.to(device)

# %% Loss and optimizer
def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

# %% Plot progress
def plot_progress(train_losses, val_losses, step):
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, color='blue', label='Training Loss')
    plt.plot(val_losses, color='red', label='Validation Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Progress')
    plt.legend()
    plt.grid(True)   
    
    # Create the "Tmp progress" folder if it doesn't exist
    if not os.path.exists("Tmp traning plots"):
       os.makedirs("Tmp traning plots")
    # Save the plot as an image in the folder
    plt.savefig(f"Tmp traning plots/loss_{step}.png")
    plt.show()
    plt.close()

# %% Validation function
def validate(model, data_loader, loss_fn):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for _, data in enumerate(data_loader, 0):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)

            outputs = model(ids, mask)
            loss = loss_fn(outputs, targets)

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss

# %% Fine tuning
losses = []
val_losses = []

def train(epoch):
    model.train()
    
    # Validation after each epoch
    val_loss = validate(model, testing_loader, loss_fn)
    print(f"Validation Loss: {val_loss}")
    
    for _,data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.float)

        outputs = model(ids, mask)

        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        losses.append(loss.item())
        val_losses.append(val_loss) # Add each step to have plot to follow
        
        # Update to console and plot
        if _%1==0:
            print(f'Step: {_}, Epoch: {epoch}, Loss:  {loss.item()}')
            plot_progress(losses, val_losses, step=_)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


for epoch in range(EPOCHS):
    train(epoch)









