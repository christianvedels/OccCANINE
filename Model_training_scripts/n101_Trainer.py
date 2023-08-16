# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 13:33:15 2023

@author: chris
"""

# Set path to file path
import os
script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)

# %% Libraries
import torch
from sklearn.metrics import accuracy_score
from torch import nn
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

#%% Function for a single training iteration
def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, verbose = False):
    model = model.train()
    losses = []
    correct_predictions = 0
    
    if verbose:
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
        # breakpoint()
        test = accuracy_score(predicted_labels.cpu(), targets.cpu())
        correct_predictions += test
        
        losses.append(loss.item())
        
        # Backward prop
        loss.backward()
        
        # Gradient Descent
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        if verbose:
            print(":",end = "")
     
    if verbose:
        print("]")
    
    return correct_predictions / len(data_loader), np.mean(losses)

# %% Eval model
def eval_model(model, data_loader, loss_fn, device):
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
            test = accuracy_score(predicted_labels.cpu(), targets.cpu())
            correct_predictions += test
            
    return correct_predictions / len(data_loader), np.mean(losses)

# %% Plot training
def plot_progress(train_losses, val_losses, step, reference_loss, model_name):
    
    plt.figure(figsize = (8, 6))
    plt.plot(train_losses, color = 'blue', label = 'Training Loss')
    plt.plot(val_losses, color = 'red', label = 'Validation Loss')
    plt.axhline(y = reference_loss, color = 'black', linestyle = 'dotted', label = 'Reference loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Progress')
    plt.legend()
    plt.grid(True)   
    
    # Create the "Tmp progress" folder if it doesn't exist
    if not os.path.exists("../Tmp traning plots"):
       os.makedirs("../Tmp traning plots")
    # Save the plot as an image in the folder
    plt.savefig("../Tmp traning plots/"+model_name+".png")
    plt.show()
    plt.close()

# %% Trainer loop
def trainer_loop(
        model, 
        epochs, 
        model_name,
        data,
        loss_fn,
        reference_loss,
        optimizer,
        device,
        scheduler,
        verbose = False
        ):
    history = defaultdict(list)
    best_accuracy = 0
    attack_switch = False
    
    # Train first epoch before plotting anything
    train_epoch(model, data['data_loader_train'], loss_fn, optimizer, device, scheduler, verbose=verbose)
  
    # Training loop
    for epoch in range(epochs):
        
        # Show details 
        print("----------")
        print(f"Epoch {epoch + 1}/{epochs}")
        
        
        # Switch when below reference loss
        if(attack_switch):
            train_acc, train_loss = train_epoch(
                model, 
                data['data_loader_train'], 
                loss_fn,    
                optimizer, 
                device, 
                scheduler, 
                verbose=verbose
            )
        else:
            train_acc, train_loss = train_epoch(
                model, 
                data['data_loader_train'], 
                loss_fn, 
                optimizer, 
                device, 
                scheduler, 
                verbose=verbose
                )
        
        print(f"Train loss {train_loss}, accuracy {train_acc}")    
        
        if(train_loss < 0.5*reference_loss):
            attack_switch = True
            print("-----> SWITCHED TO DATA LOADER WITH ATTACK")
            
        # Get model performance (accuracy and loss)
        val_acc, val_loss = eval_model(
            model,
            data['data_loader_val'],
            loss_fn,
            device,
        )
        
        print(f"Val   loss {val_loss}, accuracy {val_acc}")
        
        print(f"Reference loss: {reference_loss}")
        
        # Store stats to history
        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
        
        # Make plot
        plot_progress(
            history['train_loss'], 
            history['val_loss'], 
            step=0, 
            reference_loss = reference_loss,
            model_name = model_name
            )

        # If we beat prev performance
        if val_acc > best_accuracy:
            torch.save(
                model.state_dict(), 
                '../Trained_models/'+model_name+'.bin'
                )
            best_accuracy = val_acc
        
    return model, history
    