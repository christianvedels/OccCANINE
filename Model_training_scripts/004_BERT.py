# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 12:10:16 2023

**Reference loss:**
A problem is that a good local minimum is to guess for the probability of each 
class. The reference loss reflects this. It is the loss obtained from simply 
guessing the frequency of each of the classes. We want our network to not be 
stuck here. If the loss reamins around the level of the reference loss, then it
indicates that it is stuck in that local minimum. 

**Downsampling:**
Another problem is unbalancedness. This is mostly a problem for the HISCO code '-1',
which encodes 'no occupation'. E.g. for Danish census data this is around 70% of the data. 
This causes guessing '-1' to be a strong local minimum. This is adressed by downsampling
this category such that it represents an equal number of observations as the second most
frequent category.

**Upsampling:**
Equally there are many labels which are incredibily rare. This causes any 
learning in these labels to not be rewarded with any discernable probability. 
As such to contributes to the local minimum problem. All rare labels are
upsampled to 'UPSAMPLE_MINIMUM'.

**Attakcer()**
This function 'attacks' the text in the spirit of but much simpler than the 
TextAttack library (https://textattack.readthedocs.io/en/latest/).
The function randomly changes letters according to an 'alt_probability'.
The function also randomly inserts a word in a random location in each string, 
with the same 'alt_probability'. The words are drawn from the distribution
of words in the training data. 

"""
#%% Hyperparameters

# Which training data is used for the model
# model_domain = "HSN_DATABASE"
# model_domain = "DK_CENSUS"
model_domain = "EN_MARR_CERT"

# Parameters
sample_size = 4 # 10 to the power of this is used for training
EPOCHS = 10000
BATCH_SIZE = 128
LEARNING_RATE = 2e-3
UPSAMPLE_MINIMUM = 10000

# %% BERT finetune based on model_domain
if(model_domain == "DK_CENSUS"):
    MDL = 'Maltehb/danish-bert-botxo' # https://huggingface.co/Maltehb/danish-bert-botxo
elif(model_domain == "EN_MARR_CERT"):
    MDL = "bert-base-uncased" 
elif(model_domain == "HSN_DATABASE"):
    MDL = "GroNLP/bert-base-dutch-cased"
else:
    raise Exception("This is not implemented yet")


# %%
# Import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
from collections import defaultdict
from textwrap import wrap
import random as r

# Torch ML libraries
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

# Misc.
import string

# %%
import os
if os.environ['COMPUTERNAME'] == 'SAM-126260':
    os.chdir('D:/Dropbox/PhD/HISCO clean')
elif os.environ['COMPUTERNAME'] == 'DESKTOP-7BUTU8I':
    os.chdir('C:/Users/chris/Dropbox/PhD/HISCO clean')
else:
    raise Exception("Please define dir for this computer")
print(os.getcwd())  


# %% 
# Set intial variables and constants
# %config InlineBackend.figure_format='retina'

# Graph Designs
sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 8

# Random seed for reproducibilty
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Set GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"

# %%
# Loads given domain 
if(model_domain == "DK_CENSUS"):
    fname = "Data/Training_data/DK_census_train.csv"
elif(model_domain == "EN_MARR_CERT"):
    fname = "Data/Training_data/EN_marr_cert_train.csv" 
elif(model_domain == "HSN_DATABASE"):
    fname = "Data/Training_data/HSN_database_train.csv"
else:
    raise Exception("This is not implemented yet")

df = pd.read_csv(fname, encoding = "UTF-8")

# Handle na strings
df['occ1'] = df['occ1'].apply(lambda val: " " if pd.isna(val) else val)

# Key
key = pd.read_csv("Data/Key.csv") # Load key and convert to dictionary
key = key[1:]
key = zip(key.code, key.hisco)
key = list(key)
key = dict(key)
label2id = key
id2label = {v: k for k, v in label2id.items()}

# %% Downsapmling "no occupation"
# Find the largest category
category_counts = df['code1'].value_counts()

# Make plot
sns.distplot(category_counts.tolist())
plt.xlabel('Labels count (log scale)')
plt.xscale('log')
plt.show()

# Downsample
largest_category = category_counts.index[0]
second_largest_size = next_largest_cat = category_counts.tolist()[1]

# Split df into df with the largest category and df with other categories
df_largest = df[df.code1 == largest_category]
df_other = df[df.code1 != largest_category]

# Downsample to size of next largest cat if it is the largest
df_largest = df_largest.sample(second_largest_size, random_state=20)

# Merge 'df_noocc' and 'df_occ' into 'df' and shuffle data
df = pd.concat([df_largest, df_other], ignore_index=True)
df = df.sample(frac=1, random_state=20)  # Shuffle the rows

# Print new counts
category_counts = df['code1'].value_counts() 
print(category_counts)
    
    
# %% Upsampling the remaining data
# Labels with less than UPSAMPLE_MINIMUM have an UPSAMPLE_MINIMUM observations
# added to the data
df_original = df.copy()

# Initialize an empty DataFrame to store upsampled samples
upsampled_df = pd.DataFrame()

# Loop through unique classes (excluding the 'no occupation' class)
for class_label in df['code1'].unique():
    class_samples = df[df['code1'] == class_label]
    if(class_samples.shape[0]==0):
        continue
    if(class_samples.shape[0]<UPSAMPLE_MINIMUM):
        print(f"Upsampling: {class_samples.shape[0]} --> {UPSAMPLE_MINIMUM+class_samples.shape[0]}")
        oversampled_samples = class_samples.sample(UPSAMPLE_MINIMUM, replace=True, random_state=20)
        upsampled_df = pd.concat([upsampled_df, oversampled_samples], ignore_index=True)

# Combine upsampled data with 'no occupation' downsampled data
df = pd.concat([df, upsampled_df], ignore_index=True)
df = df.sample(frac=1, random_state=20)  # Shuffle the rows again

# Print new counts after upsampling
category_counts = df['code1'].value_counts() 
print(category_counts)

# Make plot
sns.distplot(category_counts.tolist())
plt.xlabel('Labels count (log scale)')
plt.xscale('log')
plt.show()

# %%
# Subset to smaller
if(10**sample_size < df.shape[0]):
    r.seed(20)
    df = df.sample(10**sample_size, random_state=20)
    
# %%
# Create a count plot
sns.countplot(data=df, x='code1')
plt.xlabel('Occupations')
plt.ylabel('Count')
plt.title('Distribution of Occupations')
plt.show()

# %%
# Build a BERT based tokenizer
tokenizer = BertTokenizer.from_pretrained(MDL)

# Some of the common BERT tokens
print(tokenizer.sep_token, tokenizer.sep_token_id) # marker for ending of a sentence
print(tokenizer.cls_token, tokenizer.cls_token_id) # start of each sentence, so BERT knows we’re doing classification
print(tokenizer.pad_token, tokenizer.pad_token_id) # special token for padding
print(tokenizer.unk_token, tokenizer.unk_token_id) # tokens not found in training set 

# Store length of each review 
token_lens = []

# Iterate through the content slide
for txt in df.occ1:
    tokens = tokenizer.encode(txt, max_length=512)
    token_lens.append(len(tokens))
    
# plot the distribution of occ description length
sns.distplot(token_lens)
plt.xlim([0, 256]);
plt.xlabel('Token count')

MAX_LEN = 50

# %% Labels to bin function
def labels_to_bin(df, max_value):
    df_codes = df[["code1", "code2", "code3", "code4", "code5"]]

    # Binarize
    labels_list = df_codes.values.tolist()

    # == Build outcome matrix ==
    # Construct the NxK matrix
    N = len(labels_list)
    K = int(max_value)
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


# %% Naive loss
# Step 1: Calculate the probabilities for each class (frequency of occurrence)
labels = labels_to_bin(df, max(df.code1)+1)
probs = np.mean(labels, axis=0)

# Step 2: Calculate the binary cross entropy for each class with epsilon to avoid divide by zero and log of zero
epsilon = 1e-9
probs = np.clip(probs, epsilon, 1 - epsilon)  # Clip probabilities to be in range [epsilon, 1-epsilon]

# BCE formula: -y * log(p) - (1-y) * log(1-p)
bce_per_class = -(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))

# Step 3: Sum up the binary cross entropy values for all classes
total_bce = np.mean(bce_per_class)

print("Binary Cross Entropy when guessing frequencies:", total_bce)

# %% Attacker

# List of unique words
all_text = ' '.join(df_original['occ1'].tolist())
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

#%% Dataset
class OCCDataset(Dataset):
    # Constructor Function 
    def __init__(self, occ1, df, tokenizer, max_len, n_classes, alt_prob = 0, insert_words = False):
        self.occ1 = occ1
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.targets = labels_to_bin(df, n_classes)
        self.alt_prob = alt_prob # Probability of text alteration in Attacker()
        self.insert_words = insert_words # Should random word insertation occur in Attacker()
    
    # Length magic method
    def __len__(self):
        return len(self.occ1)
    
    # get item magic method
    def __getitem__(self, item):
        occ1 = str(self.occ1[item])
        target = self.targets[item]
        
        # Implement Attack() here
        occ1 = Attacker(occ1, alt_prob = self.alt_prob, insert_words = self.insert_words)
        
        # Encoded format to be returned 
        encoding = self.tokenizer.encode_plus(
            occ1,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'occ1': occ1,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }

# %%
df_train, df_test = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)
df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED)

print(df_train.shape, df_val.shape, df_test.shape)

# %% Dataloader
def create_data_loader(df, tokenizer, max_len, batch_size,n_classes, alt_prob = 0, insert_words = False):
    ds = OCCDataset(
        occ1=df.occ1.to_numpy(),
        df=df,
        tokenizer=tokenizer,
        max_len=max_len,
        n_classes=n_classes,
        alt_prob = alt_prob, 
        insert_words = insert_words
    )
    
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=0
    )

# %%
# Create train, test and val data loaders
N_CLASSES = max(df.code1)+1
train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE, N_CLASSES, alt_prob = 0.2, insert_words = True)
val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE, N_CLASSES)
test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE, N_CLASSES)

# %%
# Examples 
data = next(iter(train_data_loader))
print(data.keys())

print(data['occ1'])
print(data['input_ids'].shape)
print(data['attention_mask'].shape)
print(data['targets'].shape)
print(data['targets'][0])

# %%
# Load the basic BERT model 
bert_model = BertModel.from_pretrained(MDL)

# %%
# Build the Sentiment Classifier class 
class OccupationClassifier(nn.Module):
    
    # Constructor class 
    def __init__(self, n_classes):
        super(OccupationClassifier, self).__init__()
        self.bert = bert_model
        self.drop = nn.Dropout(p=0.5)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
    
    # Forward propagaion class
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
          input_ids=input_ids,
          attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output
        
        #  Add a dropout layer 
        output = self.drop(pooled_output)
        return self.out(output)
    
# %%
# Instantiate the model and move to classifier
model = OccupationClassifier(int(N_CLASSES))
model = model.to(device)

# %%
# Number of hidden units
print(bert_model.config.hidden_size)

# %% 
# Optimizer Adam 
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, correct_bias=False)

total_steps = len(train_data_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

# Set the loss function 
loss_fn = nn.BCEWithLogitsLoss().to(device)

# %%
# Function for a single training iteration
def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0
    
    print("Training:", end = " ")
    
    for batch_idx, d in enumerate(data_loader):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d['targets'].to(device, dtype=torch.float)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        loss = loss_fn(outputs, targets)
        
        # Calculate correct predictions using threshold
        preds = torch.sigmoid(outputs)
        threshold = 0.5  # You can adjust this threshold based on your use case
        predicted_labels = (preds > threshold).float()
        test = Mult_accuracy(predicted_labels, targets)
        correct_predictions += test
        
        losses.append(loss.item())
        
        # Backward prop
        loss.backward()
        
        # Gradient Descent
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        print(":",end = "")
        
    print("]")
    
    return correct_predictions / n_examples, np.mean(losses)

# %%
def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    
    losses = []
    correct_predictions = 0
    
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d['targets'].to(device, dtype=torch.float)
            
            # Get model ouptuts
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            loss = loss_fn(outputs, targets)
            losses.append(loss.item())
            
            # Calculate correct predictions using threshold
            preds = torch.sigmoid(outputs)
            threshold = 0.5  # You can adjust this threshold based on your use case
            predicted_labels = (preds > threshold).float()
            test = Mult_accuracy(predicted_labels, targets)
            correct_predictions += test
            
            
            
    return correct_predictions / n_examples, np.mean(losses)

# %% Plot training
def plot_progress(train_losses, val_losses, step):
    
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, color='blue', label='Training Loss')
    plt.plot(val_losses, color='red', label='Validation Loss')
    plt.axhline(y=total_bce, color='black', linestyle='dotted', label='Reference loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Progress')
    plt.legend()
    plt.grid(True)   
    
    # Create the "Tmp progress" folder if it doesn't exist
    if not os.path.exists("Tmp traning plots"):
       os.makedirs("Tmp traning plots")
    # Save the plot as an image in the folder
    plt.savefig(f"Tmp traning plots/loss_BERT_{model_domain}_sample_size_{sample_size}.png")
    plt.show()
    plt.close()

# %% mult_accuracy
# Mult_accuracy

def Mult_accuracy(y_pred, y_true):
    y_pred = y_pred.cpu().numpy()
    y_pred = [np.where(row == 1)[0].tolist() for row in y_pred]
    y_true = y_true.cpu().numpy()
    y_true = [np.where(row == 1)[0].tolist() for row in y_true]
    
    equal_lists = [inner_list1 == inner_list2 for inner_list1, inner_list2 in zip(y_pred, y_true)]
    res = np.sum(equal_lists)
    
    return res # Returns the number of correctly predicted sets of labels

# %% Train

history = defaultdict(list)
best_accuracy = 0

# Train first epoch
train_acc, train_loss = train_epoch(
    model,
    train_data_loader,
    loss_fn,
    optimizer,
    device,
    scheduler,
    len(df_train)
)

# Training loop
for epoch in range(EPOCHS):
    
    # Show details 
    print("----------")
    print(f"Epoch {epoch + 1}/{EPOCHS}")
        
    train_acc, train_loss = train_epoch(
        model,
        train_data_loader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        len(df_train)
    )
    
    print(f"Train loss {train_loss}, accuracy {train_acc}")
    
    # Get model performance (accuracy and loss)
    val_acc, val_loss = eval_model(
        model,
        val_data_loader,
        loss_fn,
        device,
        len(df_val)
    )
    
    print(f"Val   loss {val_loss}, accuracy {val_acc}")
    
    print(f"Reference loss: {total_bce}")
    
    
    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc)
    history['val_loss'].append(val_loss)
    
    # Make plot
    plot_progress(history['train_loss'], history['val_loss'], step=0)
      
    
    # If we beat prev performance
    if val_acc > best_accuracy:
        torch.save(
            model.state_dict(), 
            f'Trained_models/BERT_{model_domain}_sample_size_{sample_size}.bin'
            )
        best_accuracy = val_acc
        
# %%
# Plot training and validation accuracy

# history['train_acc'] = [tensor.cpu().item() for tensor in history['train_acc']]
# history['val_acc'] = [tensor.cpu().item() for tensor in history['val_acc']]

plt.plot(history['train_acc'], label='train accuracy')
plt.plot(history['val_acc'], label='validation accuracy')

# Graph chars
plt.title('Training history')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.ylim([0, 1]);

# %% Model evaluiation
test_acc, _ = eval_model(
  model,
  test_data_loader,
  loss_fn,
  device,
  len(df_test)
)

test_acc.item()

# %%
def get_predictions(model, data_loader):
    model = model.eval()

    review_texts = []
    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
        for d in data_loader:
            texts = d["occ1"]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            # Get outouts
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            preds = torch.sigmoid(outputs)
            threshold = 0.5  # You can adjust this threshold based on your use case
            predicted_labels = (preds > threshold).float()

            review_texts.extend(texts)
            predictions.extend(predicted_labels)
            prediction_probs.extend(outputs)
            real_values.extend(targets)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()

    return review_texts, predictions, prediction_probs, real_values

# %%
y_occ_texts, y_pred, y_pred_probs, y_test = get_predictions(
    model,
    test_data_loader
)

#%%
report = classification_report(y_test, y_pred, output_dict=True)
report_filtered = {key: value for key, value in report.items() if not all(v == 0 for v in value.values())}

for key, value in report_filtered.items():
    print(f"{key}: {value}")
