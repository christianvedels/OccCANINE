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
import torch as t
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import plotly.express as px

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
    sample_size = 4 # SAMPLE_SIZE
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

# %% Load raw model
# Define the path to the saved binary file
model_path = '../Trained_models/'+MODEL_NAME+'.bin'

if MODEL_DOMAIN == "Multilingual":
    model_raw = XMLRoBERTaOccupationClassifier(
        n_classes = len(key), 
        model_domain = MODEL_DOMAIN, 
        tokenizer = tokenizer, 
        dropout_rate = DROPOUT_RATE
        )
else:
    model_raw = BERTOccupationClassifier(
        n_classes = len(key), 
        model_domain = MODEL_DOMAIN, 
        tokenizer = tokenizer, 
        dropout_rate = DROPOUT_RATE
    )
    
model_raw.to(device)
# Set model to evaluation mode
model_raw.eval()

# %% Get model prediction
def get_embeddings(inputs):
    embeddings = []
    BATCH_SIZE0 = BATCH_SIZE*2 # Prediction can handle larger batch size
    
    total_batches = (len(inputs) + BATCH_SIZE0 - 1) // BATCH_SIZE0  # Calculate the total number of batches

    for batch_num, i in enumerate(range(0, len(inputs), BATCH_SIZE0), 1):

        batch_inputs = inputs[i:i + BATCH_SIZE0]

        # Tokenize the batch of inputs
        batch_tokenized = tokenizer(batch_inputs, padding=True, truncation=True, return_tensors='pt')

        batch_input_ids = batch_tokenized['input_ids'].to(device)
        batch_attention_mask = batch_tokenized['attention_mask'].to(device)

        with torch.no_grad():
            output = model_best.basemodel(batch_input_ids, batch_attention_mask)
        last_hidden_state = output.last_hidden_state
        
        occupation_embedding = torch.mean(last_hidden_state, dim=1).cpu().numpy()
        embeddings.append(occupation_embedding)

        if batch_num % 10 == 0:
            print(f"Processed batch {batch_num} out of {total_batches} batches")
        
    embeddings = np.vstack(embeddings)
    
    return embeddings

def get_embeddings_raw(inputs):
    # Embeddings without training
    embeddings = []
    BATCH_SIZE0 = BATCH_SIZE*2 # Prediction can handle larger batch size
    
    total_batches = (len(inputs) + BATCH_SIZE0 - 1) // BATCH_SIZE0  # Calculate the total number of batches

    for batch_num, i in enumerate(range(0, len(inputs), BATCH_SIZE0), 1):

        batch_inputs = inputs[i:i + BATCH_SIZE0]

        # Tokenize the batch of inputs
        batch_tokenized = tokenizer(batch_inputs, padding=True, truncation=True, return_tensors='pt')

        batch_input_ids = batch_tokenized['input_ids'].to(device)
        batch_attention_mask = batch_tokenized['attention_mask'].to(device)

        with torch.no_grad():
            output = model_raw.basemodel(batch_input_ids, batch_attention_mask)
        last_hidden_state = output.last_hidden_state
        
        occupation_embedding = torch.mean(last_hidden_state, dim=1).cpu().numpy()
        embeddings.append(occupation_embedding)

        if batch_num % 10 == 0:
            print(f"Processed batch {batch_num} out of {total_batches} batches")
        
    embeddings = np.vstack(embeddings)
    
    return embeddings

    
# %% Run it
x0 = get_embeddings(
    inputs=df['concat_string0'].tolist()
    )

x1 = get_embeddings(
    inputs=df['concat_string1'].tolist()
    )


x0_raw = get_embeddings_raw(
    inputs=df['concat_string0'].tolist()
    )

x1_raw = get_embeddings_raw(
    inputs=df['concat_string1'].tolist()
    )



# %% Showing simple illustration
# Perform t-SNE to reduce dimensionality
# Reducing 768 dimensions
def tnseit(x, the_type = "no_lang"):
    tsne = TSNE(n_components=2, perplexity=30, n_iter=10000)
    tsne_results = tsne.fit_transform(x)

    # Use the Elbow Method to find the optimal number of clusters
    distortions = []
    K_range = range(1, 50)  # You can adjust the range as needed

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(tsne_results)
        distortions.append(kmeans.inertia_)

    # Find the optimal number of clusters using the Elbow Method
    optimal_num_clusters = 1  # Initialize with 1 cluster
    for i in range(1, len(distortions)):
        if (distortions[i - 1] - distortions[i]) / distortions[i - 1] < 0.1:  # You can adjust the threshold (0.1) as needed
            optimal_num_clusters = i + 1
            break

    # Perform K-Means clustering
    num_clusters = optimal_num_clusters  # Adjust this based on your needs
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(tsne_results)
    cluster_labels = kmeans.labels_

    # Create a scatter plot with clusters
    plt.figure(figsize=(10, 8))
    for cluster_id in range(num_clusters):
        plt.scatter(
            tsne_results[cluster_labels == cluster_id][:, 0], 
            tsne_results[cluster_labels == cluster_id][:, 1], 
            label=f"Cluster {cluster_id}",
            alpha = 0.1
            )

    plt.legend()
    plt.title("t-SNE Visualization with K-Means Clustering (Optimal Clusters)")

    plt.savefig(f"../Project_dissemination/Plots/tSNE_KMeans_Clustering_{the_type}.png", dpi=300)
    plt.show()
    
tnseit(x0, "with_lang")
tnseit(x1, "no_lang")
tnseit(x0_raw, "with_lang_raw")
tnseit(x1_raw, "no_lang_raw")
    







