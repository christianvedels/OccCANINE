# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 09:58:40 2023

https://huggingface.co/papluca/xlm-roberta-base-language-detection
https://colab.research.google.com/drive/15LJTckS6gU3RQOmjLqxVNBmbsBdnUEvl?usp=sharing#scrollTo=V_gbHRmNHEWU

https://github.com/abhimishra91/transformers-tutorials/blob/master/transformers_multi_label_classification.ipynb
https://colab.research.google.com/drive/1U7SX7jNYsNQG5BY1xEQQHu48Pn6Vgnyt?usp=sharing

@author: christian-vs
"""
# %% Key vars
training_data = "DK_CENSUS"
sample_size = 5 # 10 to the power of this
MAX_LEN = 200
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 100
LEARNING_RATE = 1e-05

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
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import string

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

if(training_data == "DK_CENSUS"):
    fname = "Data/Training_data/DK_census_train.csv"
else:
    raise Exception("This is not implemented yet")

df = pd.read_csv(fname, encoding = "UTF-8")

key = pd.read_csv("Data/Key.csv") # Load key and convert to dictionary
key = key[1:]
key = zip(key.code, key.hisco)
key = list(key)
key = dict(key)
label2id = key
id2label = {v: k for k, v in label2id.items()}
 
# %% Downsapmling "no occupation"
# Reduce no occ rows
# A large share is 'no occupation' this presents a balancing problem
# These have the code '2'
category_counts = df['code1'].value_counts() 
print(category_counts)
next_largest_cat = category_counts.tolist()[1]

# Split df into df with occ and df wih no occ
df_noocc = df[df.code1 == 2]
df_occ = df[df.code1 != 2]

# Downsample no occ
df



# %%
# Subset to smaller
if(10**sample_size < df.shape[0]):
    r.seed(20)
    df = df.sample(10**sample_size)


# %% Test val split
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

# Now we need to update the tokenizer with words it has not seen before
# List of unique words
all_text = ' '.join(df.occ1.tolist())
words_list = all_text.split()
unique_words = set(words_list)

# Add tokens for missing words
tokenizer.add_tokens(list(unique_words))

# %% Attacker

# List of unique words
all_text = ' '.join(df['occ1'].tolist())
words_list = all_text.split()

def Attacker(x_string, alt_prob = 0.1, insert_words = True):
    
    # breakpoint()
    
    x_string = [x_string]    
    x_string_copy = x_string.copy()
    
    if(alt_prob == 0): # Then don't waste time
        return(x_string_copy)
    
    # Alter chars
    for i in range(len(x_string_copy)):
        # alt_prob probability that nothing will happen to the string
        if r.random() < alt_prob:
            continue
        
        string_i = x_string_copy[i]
       
        num_letters = len(string_i)
        num_replacements = int(num_letters * alt_prob)
        
        indices_to_replace = r.sample(range(num_letters), num_replacements)
        
        # Convert string to list of characters
        chars = list(string_i)
        
        for j in indices_to_replace:
            chars[j] =  r.choice(string.ascii_lowercase) # replace with a random letter
            
        string_i = ''.join(chars)
               
        x_string_copy[i] = string_i
        
    if insert_words:
        for i in range(len(x_string_copy)):
            if r.random() < alt_prob: # Only make this affect alt_prob of cases
                # Word list
                occ_as_word_list = x_string_copy[i].split()
                                
                # Random word
                random_word = r.choice(words_list)
                                
                # choose a random index to insert the word
                insert_index = r.randint(0, len(occ_as_word_list))

                # insert the word into the list
                occ_as_word_list.insert(insert_index, random_word)
                
                x_string_copy[i] = " ".join(occ_as_word_list)
                    
    return(x_string_copy[0][0])

# %% Defining CustomDataset 
# (https://github.com/abhimishra91/transformers-tutorials/blob/master/transformers_multi_label_classification.ipynb)

class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len, Labels, alt_prob = 0, insert_words = False):
        self.tokenizer = tokenizer
        self._data = dataframe
        self.occ1 = dataframe.occ1
        self.targets = Labels
        self.max_len = max_len
        self.alt_prob = alt_prob # Probability of text alteration in Attacker()
        self.insert_words = insert_words # Should random word insertation occur in Attacker()

    def __len__(self):
        return len(self.occ1)

    def __getitem__(self, index):
        occ1 = str(self.occ1[index])
        
        # Attack text
        occ1 = Attacker(occ1, alt_prob = self.alt_prob, insert_words = self.insert_words)
        # breakpoint()
        occ1 = " ".join(occ1[0].split())
        
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

training_set = CustomDataset(df_train, tokenizer, MAX_LEN, labels_train, alt_prob = 0.2, insert_words = True)
testing_set = CustomDataset(df_val, tokenizer, MAX_LEN, labels_val, alt_prob = 0)

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

base_model = transformers.XLMRobertaModel.from_pretrained('xlm-roberta-base')

# Adapt model size to the tokens added:
base_model.resize_token_embeddings(len(tokenizer))

# Define class
class xmlRoBERTaClass(torch.nn.Module):
    def __init__(self):
        super(xmlRoBERTaClass, self).__init__()
        self.l1 = base_model
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, len(key))
    
    def forward(self, ids, mask):
        # breakpoint()
        
        _, output_1= self.l1(ids, attention_mask = mask, return_dict=False)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output

# Make instance
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
        if _%20==0:
            print(f'Step: {_}, Epoch: {epoch}, Loss:  {loss.item()}')
            
            # Validation
            val_loss = validate(model, testing_loader, loss_fn)
            print(f"Validation Loss: {val_loss}")
            val_losses[_] = val_loss
            
            plot_progress(losses, val_losses, step=_)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


for epoch in range(EPOCHS):
    train(epoch)
    
    
# %% Validation
def validation_and_metrics():
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)
            outputs = model(ids, mask)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

    fin_outputs = np.array(fin_outputs) >= 0.5
    accuracy = metrics.accuracy_score(fin_targets, fin_outputs)
    f1_score_micro = metrics.f1_score(fin_targets, fin_outputs, average='micro')
    f1_score_macro = metrics.f1_score(fin_targets, fin_outputs, average='macro')

    print("Final Validation Metrics:")
    print(f"Accuracy Score = {accuracy}")
    print(f"F1 Score (Micro) = {f1_score_micro}")
    print(f"F1 Score (Macro) = {f1_score_macro}")

# %% Run validation
validation_and_metrics()

# %% Construct prediction function
def predict_hiscos(strings):
    x = tokenizer(
        strings,
        max_length=MAX_LEN,
        pad_to_max_length=True,
        return_tensors="pt"
    )
    ids = x["input_ids"]
    mask = x["attention_mask"]
    logits = model(ids, mask)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(logits.squeeze().cpu())
    predictions = np.zeros(probs.shape)
    predictions[np.where(probs >= 0.5)] = 1
    predicted_indices = np.where(predictions == 1)[1]
    predicted_labels = [key[i] for i in predicted_indices]
    return(predicted_labels)


# %% Run predictions
predict_hiscos(df.occ1[1:100].tolist())

    
# %% Create a directory to save the model
path = "Trained_models/XMLRoBERTa_"+training_data+"_sample_size"+str(sample_size)
path_vocab = path+"/vocab"
if not os.path.exists(path):
    os.makedirs(path)
    
if not os.path.exists(path_vocab):
    os.makedirs(path_vocab)

# Save the trained model
output_model_file = path+"/model.bin"
output_vocab_file = path_vocab

model_to_save = model
torch.save(model_to_save, output_model_file)
tokenizer.save_vocabulary(output_vocab_file)




