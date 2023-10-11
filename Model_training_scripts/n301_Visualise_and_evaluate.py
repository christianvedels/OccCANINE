# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 16:04:32 2023

@author: christian-vs
"""
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

#%% Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    

# %% Attention_Viz
def Attention_Viz(sentence, layer = 0):
    inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
        
    with torch.no_grad():
        outputs = model_best.bert(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            output_attentions=True
        )
    attention_weights = outputs.attentions

    # predictions
    # get_predictions(inputs)
    
    layer = 0  # Choose the layer to visualize (0 to num_layers - 1)
    num_heads = attention_weights[layer][0].shape[0]
    
    # Plot parameters
    rows = 3
    cols = num_heads // rows
    plt.figure(figsize=(15, 10))

    for head in range(num_heads):
        plt.subplot(rows, cols, head + 1)
        attention = attention_weights[layer][0][head].squeeze().cpu().numpy()
        plt.imshow(attention, cmap='viridis', aspect='auto')
        plt.title(f'Head {head}')
        plt.xticks(range(len(inputs['input_ids'][0])), tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]), rotation=45)
        plt.yticks(range(len(inputs['input_ids'][0])), tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]))
        if head == 0:
            plt.ylabel(f'Layer {layer}')
    
    plt.tight_layout()
    plt.show()
    
# %% Attention_Viz average
def Attention_Viz(sentence, layer = 0):
    inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
        
    with torch.no_grad():
        outputs = model_best.bert(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            output_attentions=True
        )
    attention_weights = outputs.attentions

    # predictions
    # get_predictions(inputs)
    
    layer = 0  # Choose the layer to visualize (0 to num_layers - 1)
    num_heads = attention_weights[layer][0].shape[0]
    
    # Plot parameters
    rows = 3
    cols = num_heads // rows
    plt.figure(figsize=(15, 10))

    for head in range(num_heads):
        plt.subplot(rows, cols, head + 1)
        attention = attention_weights[layer][0][head].squeeze().cpu().numpy()
        plt.imshow(attention, cmap='viridis', aspect='auto')
        plt.title(f'Head {head}')
        plt.xticks(range(len(inputs['input_ids'][0])), tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]), rotation=45)
        plt.yticks(range(len(inputs['input_ids'][0])), tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]))
        if head == 0:
            plt.ylabel(f'Layer {layer}')
    
    plt.tight_layout()
    plt.show()
    
# %%
def Attention_Viz2(sentence):
    # breakpoint()
    inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
    
    with torch.no_grad():
        outputs = model_best.bert(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            output_attentions=True
        )
    attention_weights = outputs.attentions
    
    # Calculate the mean attention score for each word across all layers
    mean_attention_scores = []
    for layer_attention in attention_weights:
        layer_mean_attention = torch.mean(layer_attention, dim=(1, 2))
        mean_attention_scores.append(layer_mean_attention)
    
    # breakpoint()
    mean_attention_scores = torch.stack(mean_attention_scores).cpu().numpy()
    mean_attention_scores = mean_attention_scores.mean(axis=0)
    
    # Convert input_ids to a NumPy array
    input_ids_np = inputs['input_ids'].numpy()
    
    # Get the tokens from the input
    tokens = tokenizer.convert_ids_to_tokens(input_ids_np[0])
    
    # Create a bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(tokens, mean_attention_scores[0])
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Tokens')
    plt.ylabel('Mean Attention Score')
    plt.title('Average Attention Score for Each Token')
    plt.tight_layout()
    
    # Save the plot
    # plt.savefig('average_attention_scores.png')
    plt.show()

# %%
import n101_Trainer as t

y_occ_texts, y_pred, y_pred_probs, y_test = t.get_predictions(
    model_best,
    data['data_loader_test'],
    device
)
report = classification_report(y_test, y_pred, output_dict=True)

print_report(report, MODEL_NAME)
    
# %% Run it
# Get example
# examples = []
# for d in data['data_loader_test']:
#     examples.extend(d['occ1'][0])
    
# df = pd.DataFrame({'occ1': examples})
    

examples = [    
    'block printer', 
    'post office official', 
    'private 2/5 lincoln reg', 
    'brass polisher', 
    'in the coast guard service', 
    'forestry commission worker', 
    'coal and provision dealer'
    ]

examples = [Concat_string(x, "unk") for x in examples]

for e in examples:
    inputs = tokenizer.encode_plus(
        e,
        add_special_tokens=True,
        padding = 'max_length',
        max_length = MAX_LEN,
        return_token_type_ids=False,
        return_attention_mask=True,
        return_tensors='pt',
        truncation = True
    )
    
    print(inputs)


input_ids = inputs['input_ids']
attention_mask = inputs["attention_mask"]
 
with torch.no_grad():
    logits = model_best(input_ids, attention_mask)
predicted_probs = torch.sigmoid(logits)
threshold = 0.5  # You can adjust this threshold based on your use case
predicted_labels = [key[i] for i, prob in enumerate(predicted_probs[0]) if prob > threshold]


attacker = AttackerClass(df)

Attention_Viz(attacker.attack(examples[5]), layer = 0)
 
Attention_Viz2('no longer active but used to be a blacksmith')
Attention_Viz2(examples[4])
