# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 13:33:15 2023

"""

# Set path to file path
import os
script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)


# MODEL_DOMAIN = "DK_CENSUS"

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
def plot_progress(train_losses, val_losses, train_acc, val_acc, reference_loss, model_name):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

    ax1.plot(train_losses, color='blue', label='Training Loss')
    ax1.plot(val_losses, color='red', label='Validation Loss')
    ax1.axhline(y=reference_loss, color='black', linestyle='dotted', label='Reference loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(train_acc, color='blue', label='Training Accuracy')
    ax2.plot(val_acc, color='red', label='Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    # Create the "Tmp progress" folder if it doesn't exist
    if not os.path.exists("../Tmp training plots"):
        os.makedirs("../Tmp training plots")
    
    # Save the plot as an image in the folder
    plt.savefig("../Tmp training plots/" + model_name + ".png")
    #plt.show()
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
        verbose = False,
        switch_attack = 0.75
        ):
    history = defaultdict(list)
    best_accuracy = 0
    attack_switch = False
    
    # # Train first epoch before plotting anything
    # print("Started training one epoch as warmup")
    # train_epoch(model, data['data_loader_train'], loss_fn, optimizer, device, scheduler, verbose=True)
  
    # Training loop
    for epoch in range(epochs):
        
        # Show details 
        print("----------")
        print(f"Epoch {epoch + 1}/{epochs}")
        
        
        # Switch when below reference loss
        if(attack_switch):
            train_acc, train_loss = train_epoch(
                model, 
                data['data_loader_train_attack'], 
                loss_fn,    
                optimizer, 
                device, 
                scheduler, 
                verbose=True
            )
        else:
            train_acc, train_loss = train_epoch(
                model, 
                data['data_loader_train'], 
                loss_fn, 
                optimizer, 
                device, 
                scheduler, 
                verbose=True
                )
        
        print(f"Train loss {train_loss}, accuracy {train_acc}")    
        
        if(train_loss < switch_attack*reference_loss):
            if(not(attack_switch)):
                print("-----> SWITCHED TO DATA LOADER WITH ATTACK")
            attack_switch = True
            
            
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
            history['train_acc'],
            history['val_acc'],
            reference_loss = reference_loss,
            model_name = model_name
            )

        # If we beat prev performance
        if val_acc > best_accuracy:
            print("Saved improved model")
            torch.save(
                model.state_dict(), 
                '../Trained_models/'+model_name+'.bin'
                )
            best_accuracy = val_acc
            
            tokenizer_save_path = '../Trained_models/' + model_name + '_tokenizer'
            data['tokenizer'].save_pretrained(tokenizer_save_path)
        
    return model, history

# %%
def get_predictions(model, data_loader, device):
    model = model.eval()

    occ1 = []
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

            occ1.extend(texts)
            predictions.extend(predicted_labels)
            prediction_probs.extend(outputs)
            real_values.extend(targets)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()

    return occ1, predictions, prediction_probs, real_values
    
# %% Print report
# Convert the report dictionary to a string
def print_report(report, model_name):
    # Filter report to remove zero entries (zero support)
    report = {key: value for key, value in report.items() if not all(v == 0 for v in value.values())}
    # breakpoint()
    report_str = ""
    for label, metrics in report.items():
        if label == 'accuracy':
            report_str += f"Accuracy: {metrics}\n"
        else:
            report_str += f"Label: {label}\n"
            for metric_name, metric_value in metrics.items():
                report_str += f"{metric_name}: {metric_value}\n"
            report_str += "\n"
    
    # Specify the file path
    file_path = "../Tmp training plots/Classification_report" + model_name + ".txt"
    
    # Write the report string to the file
    with open(file_path, 'w') as file:
        file.write(report_str)

    print(f"Classification report saved to {file_path}")