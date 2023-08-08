# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 09:58:40 2023

Inpriration
https://huggingface.co/papluca/xlm-roberta-base-language-detection
https://colab.research.google.com/drive/15LJTckS6gU3RQOmjLqxVNBmbsBdnUEvl?usp=sharing#scrollTo=V_gbHRmNHEWU
https://github.com/abhimishra91/transformers-tutorials/blob/master/transformers_multi_label_classification.ipynb
https://colab.research.google.com/drive/1U7SX7jNYsNQG5BY1xEQQHu48Pn6Vgnyt?usp=sharing

**Reference loss:**
A problem is that a good local minimum is to guess for the probability of each 
class. The reference loss reflects this. It is the loss obtained from simply 
guessing the frequency of each of the classes. We want our network to not be 
stuck here. If the loss reamins around the level of the reference loss, then it
indicates that it is stuck in a local minimum.

**Downsampling:**
Another problem is unbalancedness. This is mostly a problem for the HISCO code '-1',
which encodes 'no occupation'. E.g. for Danish census data this is around 70% of the data. 
This causes guessing '-1' to be a strong local minimum. This is adressed by downsampling
this category such that it represents an equal number of observations as the second most
frequent category.

**Attakcer()**
This function 'attacks' the text in the spirit of but much simpler than the 
TextAttack library (https://textattack.readthedocs.io/en/latest/).
The function randomly changes letters according to an 'alt_probability'.
The function also randomly inserts a word in a random location in each string, 
with the same 'alt_probability'. The words are drawn from the distribution
of words in the training data. 

@author: christian-vs
"""
# %% Key vars
model_domain = "DK_CENSUS"
sample_size = 4 # 10 to the power of this
MAX_LEN = 128
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-03
PRINT_VAL_FREQ = 50 # How many steps between reports to console, validation and plots are updated

MDL = 'xlm-roberta-base' # Base model to fine tune from

toyrun = False # Reduces the labels to only 3 possible outcomes for a simpler toyrun

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

# %% Set wd()
if os.environ['COMPUTERNAME'] == 'SAM-126260':
    os.chdir('D:/Dropbox/PhD/HISCO clean')
else:
    os.chdir('C:/Users/chris/Dropbox/PhD/HISCO clean')
print(os.getcwd())  

#%% Load data
# Loads given domain 
if(model_domain == "DK_CENSUS"):
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

# Downsample to size of next largest cat
df_noocc = df_noocc.sample(next_largest_cat, random_state=20)

# Merge 'df_noocc' and 'df_occ' into 'df' and shuffle data
df = pd.concat([df_noocc, df_occ], ignore_index=True)
df = df.sample(frac=1, random_state=20)  # Shuffle the rows

# Print new counts
category_counts = df['code1'].value_counts() 
print(category_counts)

# %%
# Subset to smaller
if(10**sample_size < df.shape[0]):
    r.seed(20)
    df = df.sample(10**sample_size, random_state=20)


# %% Test val split
# Split into internal test/train/val
if(sample_size>4):
    TEST_SIZE = 10**4
else:
    TEST_SIZE = 0.1

df_train, df_test = train_test_split(df, test_size= TEST_SIZE)

df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

print(f"Train / test samples: {len(df_train)} / {len(df_test)}")

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
labels_test = labels_to_bin(df_test, max_value = max_val)

if(toyrun):
    labels_bin = labels_bin[:, :3]
    labels_train = labels_train[:, :3]
    labels_test = labels_test[:, :3]

# Add to df
df['labels'] = labels_bin.tolist()
df_train['labels'] = labels_train.tolist()
df_test['labels'] = labels_test.tolist()

# %% Naive loss
# Step 1: Calculate the probabilities for each class (frequency of occurrence)
probs = np.mean(labels_train, axis=0)

# Step 2: Calculate the binary cross entropy for each class with epsilon to avoid divide by zero and log of zero
epsilon = 1e-9
probs = np.clip(probs, epsilon, 1 - epsilon)  # Clip probabilities to be in range [epsilon, 1-epsilon]

# BCE formula: -y * log(p) - (1-y) * log(1-p)
bce_per_class = -(labels_train * np.log(probs) + (1 - labels_train) * np.log(1 - probs))

# Step 3: Sum up the binary cross entropy values for all classes
total_bce = np.mean(bce_per_class)

print("Binary Cross Entropy when guessing frequencies:", total_bce)

#%% Tokenizer updates
tokenizer = AutoTokenizer.from_pretrained(MDL)

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

    def __init__(self, dataframe, tokenizer, max_len, alt_prob = 0, insert_words = False):
        self.tokenizer = tokenizer
        self._data = dataframe
        self.occ1 = dataframe.occ1
        self.targets = dataframe.labels
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

training_set = CustomDataset(df_train, tokenizer, MAX_LEN, alt_prob = 0.2, insert_words = True)
testing_set = CustomDataset(df_test, tokenizer, MAX_LEN, alt_prob = 0)

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
base_model = transformers.XLMRobertaModel.from_pretrained(MDL)

# Adapt model size to the tokens added:
base_model.resize_token_embeddings(len(tokenizer))

# Define class
class xmlRoBERTaClass(torch.nn.Module):
    def __init__(self, heads):
        super(xmlRoBERTaClass, self).__init__()
        self.l1 = base_model
        self.l2 = torch.nn.Dropout(0.5)
        self.l3 = torch.nn.Linear(768, heads)
    
    def forward(self, ids, mask):
        # breakpoint()
        
        _, output_1= self.l1(ids, attention_mask = mask, return_dict=False)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output

# Make instance
model = xmlRoBERTaClass(heads = len(key))

if(toyrun):
    model = xmlRoBERTaClass(heads = 3)

# Freeeze base layer
# Freeze base model's parameters
for param in model.l1.parameters():
    param.requires_grad = False

# Send to device
model.to(device)

# %% Loss and optimizer
def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

# %% Plot progress
def plot_progress(train_losses, val_losses, step, phase):
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, color='blue', label='Training Loss')
    plt.plot(val_losses, color='red', label='Validation Loss')
    plt.axhline(y=total_bce, color='black', linestyle='dotted', label='Reference loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss Progress (phase {phase})')
    plt.legend()
    plt.grid(True)   
    
    # Create the "Tmp progress" folder if it doesn't exist
    if not os.path.exists("Tmp traning plots"):
       os.makedirs("Tmp traning plots")
    # Save the plot as an image in the folder
    plt.savefig(f"Tmp traning plots/loss_{model_domain}_sample_size_{sample_size}_phase_{phase}_{step}.png")
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

def train(epoch, phase):
    model.train()
    val_loss = validate(model, testing_loader, loss_fn)
    
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
        if _%PRINT_VAL_FREQ==0:
            print(f'Phase: {phase}, Step: {_}, Epoch: {epoch}, Loss:  {loss.item()}, reference: {total_bce}')
            
            # Validation
            val_loss = validate(model, testing_loader, loss_fn)
            print(f"Validation Loss: {val_loss}")
            val_losses[_] = val_loss
            
            plot_progress(losses, val_losses, step=_, phase = phase)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# %% Phase 1: Fine tune top layers
# This will generally only learn to guess for the frequency of each class
# regardless of the data it is fed. 
best_val_loss = float('inf')
patience = 3  # Number of epochs without improvement to tolerate
no_improvement_count = 0

for epoch in range(EPOCHS):
    train(epoch, phase = 1)
    val_loss = validate(model, testing_loader, loss_fn)
    print(f"Validation Loss after phase 1 epoch {epoch}: {val_loss}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improvement_count = 0
    else:
        no_improvement_count += 1
        if no_improvement_count >= patience:
            print(f"No improvement in validation loss for {patience} epochs. Phase 1 ends.")
            break
    
 # %% Phase 2: Fine-tune entire model for specified number of epochs   
# Unfreeze all parameters for the second phase
for param in model.parameters():
    param.requires_grad = True
    
# Set loss and val loss empty
losses_phase1 = losses.copy()
val_losses_phase1 = val_losses.copy()
losses = []
val_losses = []

# Set learning rate for the second phase
second_phase_lr = 1e-2 * LEARNING_RATE

# Reset optimizer for second phase
optimizer = torch.optim.AdamW(model.parameters(), lr = second_phase_lr)  # Adjust learning rate as needed

for epoch in range(EPOCHS):
    train(epoch, phase = 2)
    val_loss = validate(model, testing_loader, loss_fn)
    print(f"Validation Loss after phase 2 epoch {epoch}: {val_loss}")
    
    
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

# %% Save finetuned model
path = "Trained_models/XMLRoBERTa_"+model_domain+"_sample_size"+str(sample_size)
path_tokenizer = path+"/tokenizer"
if not os.path.exists(path):
    os.makedirs(path)
    
if not os.path.exists(path_tokenizer):
    os.makedirs(path_tokenizer)

# Save the trained model
output_model_file = path+"/model.bin"
output_tokenizer_file = path_tokenizer

model_to_save = model
torch.save(model_to_save, output_model_file)
tokenizer.save_pretrained(output_tokenizer_file)



