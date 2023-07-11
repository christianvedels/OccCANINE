# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 13:18:34 2023

@author: chris
Based on: https://medium.com/@keruchen/train-a-xlm-roberta-model-for-text-classification-on-pytorch-4ccf0b30f762
"""

#%% Libraries
import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import AutoTokenizer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#%% Detect GPU
# check if we have cuda installed
if torch.cuda.is_available():
    # to use GPU
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('GPU is:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

#%% Load data
if os.environ['COMPUTERNAME'] == 'SAM-126260':
    os.chdir('D:/Dropbox/PhD/HISCO/HISCO clean')
else:
    os.chdir('C:/Users/chris/Dropbox/PhD/HISCO clean')
print(os.getcwd())

fname = "Data/Training_data/DK_census_train.csv"
df = pd.read_csv(fname, encoding = "UTF-8", sep = ","
                 # , nrows = 10000
                 )

# Load keys
key = pd.read_csv("Data/Key.csv")

#%% Remove rare categories

# # Group by 'code1' and count occurrences
df_n = df.groupby('code1').size().reset_index(name='n')

# # Filter rows where count is greater than 1
df_n = df_n[df_n['n'] > 1]

# # Filter rows in df where 'code1' is in df_n['code1']
df = df[df['code1'].isin(df_n['code1'])]

# Convert to string
df.code1 = df.code1.astype(str)
df.code2 = df.code2.astype(str)
df.code3 = df.code3.astype(str)
df.code4 = df.code4.astype(str)
df.code5 = df.code5.astype(str)

#%% Tokenizer
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')

# Now we need to update the tokenizer with words it has not seen before
# List of unique words
all_text = ' '.join(df.occ1.tolist())
words_list = all_text.split()
unique_words = set(words_list)

# Add tokens for missing words
tokenizer.add_tokens(list(unique_words))

# Testing tokenizer
print({x : tokenizer.encode(x, add_special_tokens=False) for x in df.occ1.values[0].split()})

# Tokenize features
MAX_LEN = 128
features = df.occ1.values.tolist()
tokenized_feature = tokenizer.batch_encode_plus(
                            # Sentences to encode
                            features, 
                            # Add '[CLS]' and '[SEP]'
                            add_special_tokens = True,
                            # Add empty tokens if len(text)<MAX_LEN
                            padding = 'max_length',
                            # Truncate all sentences to max length
                            truncation=True,
                            # Set the maximum length
                            max_length = MAX_LEN, 
                            # Return attention mask
                            return_attention_mask = True,
                            # Return pytorch tensors
                            return_tensors = 'pt'       
                   )

#%% Structure target
# convert label into numeric 
target = df.code1.values.tolist()
le = LabelEncoder()
le.fit(target)
target_num = le.transform(target)

#%% Train/test split
train_inputs, validation_inputs, train_labels, validation_labels, train_masks, validation_masks = train_test_split(
    tokenized_feature['input_ids'],
    target_num,
    tokenized_feature['attention_mask'],
    random_state=20, 
    test_size=0.05, 
    stratify=target_num
    )

#%% Dataloader
# define batch_size
batch_size = 16
# Create the DataLoader for our training set
train_data = TensorDataset(train_inputs, train_masks, torch.tensor(train_labels))
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
# Create the DataLoader for our test set
validation_data = TensorDataset(validation_inputs, validation_masks, torch.tensor(validation_labels))
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)


#%% BertForSequenceClassification
from transformers import XLMRobertaForSequenceClassification, AdamW, get_linear_schedule_with_warmup
model = XLMRobertaForSequenceClassification.from_pretrained(
    "xlm-roberta-base", 
    # Specify number of classes
    num_labels = len(set(target)), 
    # Whether the model returns attentions weights
    output_attentions = False,
    # Whether the model returns all hidden-states 
    output_hidden_states = False
)

model.resize_token_embeddings(len(tokenizer))

#%% Optimizer & Learning Rate Scheduler
optimizer = AdamW(model.parameters(),
                  lr = 2e-5, 
                  eps = 1e-8 
                )

# Number of training epochs
epochs = 100000
# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * epochs
# Create the learning rate scheduler
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)

model.cuda()

#%% Train
import time
# Store the average loss after each epoch 
loss_values = []
# number of total steps for each epoch
print('total steps per epoch: ',  len(train_dataloader) / batch_size)
# looping over epochs
for epoch_i in range(0, epochs):
    
    print('training on epoch: ', epoch_i)
    # set start time 
    t0 = time.time()
    # reset total loss
    total_loss = 0
    # model in training 
    model.train()
    # loop through batch 
    for step, batch in enumerate(train_dataloader):
        # load data from dataloader 
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        # clear any previously calculated gradients 
        model.zero_grad()
        # get outputs
        outputs = model(b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask,
                        labels=b_labels)
        # get loss
        loss = outputs[0]
        
        # total loss
        total_loss += loss.item()
        # clip the norm of the gradients to 1.0.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # update optimizer
        optimizer.step()
        # update learning rate 
        scheduler.step()
        
        # Progress update every 50 step 
        if step % 50 == 0 and not step == 0:
            print('training on step: ', step, 'of')
            print('total time used is: {0:.2f} s'.format(time.time() - t0))
            print('loss: ', loss.item())
    # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(train_dataloader)
    # Store the loss value for plotting the learning curve.
    loss_values.append(avg_train_loss)
    print("-----> average training loss: {0:.2f}".format(avg_train_loss))

#%% Test
# Test
import numpy as np
t0 = time.time()
# model in validation mode
model.eval()
# save prediction
predictions,true_labels =[],[]
i = 0
# evaluate data for one epoch
for batch in validation_dataloader:
    # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)
    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask, b_labels = batch
    # validation
    with torch.no_grad():
        outputs = model(b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask)
    # get output
    logits = outputs[0]
    # move logits and labels to CPU
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    final_prediction = np.argmax(logits, axis=-1).flatten()
    predictions.append(final_prediction)
    true_labels.append(label_ids)
    print('Batch', i)
    i += 1
    
print('total time used is: {0:.2f} s'.format(time.time() - t0))

#%% Convert to labels
# convert numeric label to string
final_prediction_list = le.inverse_transform(np.concatenate(predictions))
final_truelabel_list = le.inverse_transform(np.concatenate(true_labels))

#%% Eval
from sklearn.metrics import confusion_matrix, classification_report
cr = classification_report(final_truelabel_list, 
                           final_prediction_list, 
                           output_dict=False)
print(cr)