# -*- coding: utf-8 -*-
"""
This script trains an AI to categorize HISCO codes
Strategy:
    - Model 1 guesses the hisco code
    - Model 2 guesses the number of occupations (0-5)

Based on: https://www.atmosera.com/blog/text-classification-with-neural-networks/
Multi label from: https://machinelearningmastery.com/multi-label-classification-with-deep-learning/
See: https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html

# Custom loss function: https://medium.com/@matrixB/modified-cross-entropy-loss-for-multi-label-classification-with-class-a8afede21eb9
# But this was modified with by taking the square root which ensures, that not
# too much weight is put on some incredibly rare labels. 

Created on Wed Aug 24 11:24:45 2022
@author: Christian Vedel
"""

for s in range(3, 6):
    # %% Parameters
    n_epochs = 10000 # The networks trains a maximum of these epochs
    dropout_rate = 0.5 # Base dropout rate
    alt_prob = 0.5 # Probability of text alteration (augmentation) (see Attack)
    unique_strings = False # Should the training be based on raw data or only unique strings?
     
    #################################
    chars = "occ1"
    hisco_level = 5
    #################################
    
    batch_size = 2**10
    sample_size = s # 10^s
    
    if(batch_size>(10**sample_size)):
        batch_size = 10**sample_size
        
    # %% Dynamic memory aloc. (does not work?)
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    
    
    # %% Import packages
    import matplotlib.pyplot as plt
    import os
    import tensorflow as tf
    import pandas as pd
    import numpy as np
    from tensorflow import keras
    import json
    from keras_preprocessing.text import tokenizer_from_json
    import random as r
    import string
    
    # Test if GPU is found
    if tf.test.gpu_device_name(): 
        print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")
    
    #%% Load data
    if os.environ['COMPUTERNAME'] == 'SAM-126260':
        os.chdir('D:/Dropbox/PhD/HISCO')
    else:
        os.chdir('C:/Users/chris/Dropbox/PhD/HISCO')
    print(os.getcwd())
    
    fname = "Data/HISCO_all"+str(hisco_level)+".csv"
    df = pd.read_csv(fname, encoding = "UTF-8")
    
    # fname = "Data/Toydata"+str(hisco_level)+".csv"
    # df = pd.read_csv(fname , encoding = "UTF-8")
    
    # Subset to only train
    df = df[df["train"]==1]
    
    # Subset to only data with occ info
    df = df[df.no_occ != 1]
    
    # Load keys
    key = pd.read_csv("Data/key5.csv")
    
    # If sample size is too large
    if(10**sample_size>df.shape[0]):
        print("Sample size larger than data. Used all data.")
    else:
        # Subset to larger toy data
        r.seed(20)
        df = df.sample(10**sample_size)
    
    # %% Attacker
    
    # List of unique words
    all_text = ' '.join(df[chars].tolist())
    words_list = all_text.split()
    # unique_words = list(set(words_list))
    
    # If too few words in e.g. UK marriage certificates then use wiki dump
    
    def Attacker(x_string, alt_prob = 0.1, insert_words = True):
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
                    word_list = x_string_copy[i].split()
                                    
                    # Random word
                    random_word = r.choice(word_list)
                                    
                    # choose a random index to insert the word
                    insert_index = r.randint(0, len(word_list))
    
                    # insert the word into the list
                    word_list.insert(insert_index, random_word)
                    
                    x_string_copy[i] = " ".join(word_list)
                        
        # print(x_string_copy)           
            
                    
        # res = pd.DataFrame(x_string_copy)
        
        return(x_string_copy)
    
    #%% Sequential models
    # Params copy pasted from '01_Data_prep.py'
    max_words = 50000
    
    # Layer import
    from keras.models import Sequential
    from keras.layers import Dense, Flatten, Embedding, LSTM, Dropout, Bidirectional, GlobalAveragePooling1D, LeakyReLU, TextVectorization, Lambda
    from tensorflow.keras import regularizers, Input, Model
    
    # Model 1:
    # n_outputs: Number of labels
    # max_length: Max string length
    # layer_size: Base size of layers
    # to_compile: Should the model be compiled?
    
    def make_model1(n_outputs, max_length, vectorize_layer, layer_size = 128, to_compile = True):
        forward_layer = LSTM(layer_size, return_sequences=True)
        backward_layer = LSTM(layer_size, return_sequences=True, go_backwards=True)
            
        model = Sequential() 
        model.add(Input(shape=(1,), dtype=tf.string))
        model.add(vectorize_layer)
        model.add(Embedding(max_words, int(layer_size/4), input_length=max_length))
        model.add(Dropout(dropout_rate/4))
        model.add(Bidirectional(
            forward_layer, backward_layer=backward_layer,
            input_shape = (int(layer_size/2), layer_size)
            ))
        model.add(GlobalAveragePooling1D())
        model.add(Dropout(0.5))
        model.add(Dense(
            layer_size*2, 
            activation=LeakyReLU(alpha=0.01)
            ))
        model.add(Dropout(0.5))
        model.add(Dense(
            layer_size, 
            activation=LeakyReLU(alpha=0.01)
            ))
        model.add(Dense(n_outputs, activation='sigmoid', name='multi_o/p')) # Because we want probs to be more than 1
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', f1_m, precision_m, recall_m]) 
        model.summary()
        return model
    
    # %% Plot callback
    # From: https://medium.com/geekculture/how-to-plot-model-loss-while-training-in-tensorflow-9fa1a1875a5
    from IPython.display import clear_output
    
    class PlotLearning1(keras.callbacks.Callback):
        """
        Callback to plot the learning curves of the model during training.
        """
        def on_train_begin(self, logs={}):
            self.metrics = {}
            for metric in logs:
                self.metrics[metric] = []
                
    
        def on_epoch_end(self, epoch, logs={}):
            # Storing metrics
            for metric in logs:
                if metric in self.metrics:
                    self.metrics[metric].append(logs.get(metric))
                else:
                    self.metrics[metric] = [logs.get(metric)]
            
            # Plotting
            metrics = [x for x in logs if 'val' not in x]
            
            f, axs = plt.subplots(1, len(metrics), figsize=(15,5))
            clear_output(wait=True)
    
            for i, metric in enumerate(metrics):
                axs[i].plot(range(1, epoch + 2), 
                            self.metrics[metric], 
                            label=metric)
                if logs['val_' + metric]:
                    axs[i].plot(range(1, epoch + 2), 
                                self.metrics['val_' + metric], 
                                label='val_' + metric)
                    
                axs[i].legend()
                axs[i].grid()
    
            plt.tight_layout()
            plt.savefig("Plots/Mod_char_aug"+str(alt_prob)+"_unique_"+str(unique_strings)+"_smplsize_"+str(sample_size)+"_lvl"+str(hisco_level)+".png")
            # plt.close()
            plt.show()
            
    class PlotLearning2(keras.callbacks.Callback):
        """
        Callback to plot the learning curves of the model during training.
        """
        def on_train_begin(self, logs={}):
            self.metrics = {}
            for metric in logs:
                self.metrics[metric] = []
                
    
        def on_epoch_end(self, epoch, logs={}):
            # Storing metrics
            for metric in logs:
                if metric in self.metrics:
                    self.metrics[metric].append(logs.get(metric))
                else:
                    self.metrics[metric] = [logs.get(metric)]
            
            # Plotting
            metrics = [x for x in logs if 'val' not in x]
            
            f, axs = plt.subplots(1, len(metrics), figsize=(15,5))
            clear_output(wait=True)
    
            for i, metric in enumerate(metrics):
                axs[i].plot(range(1, epoch + 2), 
                            self.metrics[metric], 
                            label=metric)
                if logs['val_' + metric]:
                    axs[i].plot(range(1, epoch + 2), 
                                self.metrics['val_' + metric], 
                                label='val_' + metric)
                    
                axs[i].legend()
                axs[i].grid()
    
            plt.tight_layout()
            plt.savefig("Plots/Mod_char_warmup_aug"+str(alt_prob)+"_unique_"+str(unique_strings)+"_smplsize_"+str(sample_size)+"_lvl"+str(hisco_level)+".png")
            # plt.close()
            plt.show()
            
    
    plt_callback1 = PlotLearning1()
    plt_callback2 = PlotLearning2()
    
    # %% Callbacks
    
    patience0 = 10**(7-sample_size)
    if(s>6):
        patience0 = 10
    
    # Callback to stop when covergence
    from keras.callbacks import EarlyStopping
    callback2 = EarlyStopping(
        monitor="val_loss",
        min_delta = 0.0000001,
        patience = patience0,
        restore_best_weights=True
        )
    
    
    # %% Labels to matrix
    # def y_to_mat(df, num_classes):
    #     # Subset labels
    #     df0 = df
    
    #     # labels
    #     y0 = df0[["code1"]]
    #     y0 = y0.reset_index()
    #     y0 = tf.keras.utils.to_categorical(y0, num_classes = num_classes)
    
    #     for i in range(3): # Note that 5th column was dropped. Almost never used
    #         y_i = df0[["code"+str(i+2)]]    
    #         y_i = tf.keras.utils.to_categorical(y_i, num_classes = num_classes)
    #         n_pads = y0.shape[1] - y_i.shape[1]
    #         y_i = np.pad(y_i, pad_width=[(0, 0), (0, n_pads)], mode = "constant") # Explained here https://stackoverflow.com/questions/35751306/python-how-to-pad-numpy-array-with-zeros
    #         # Add to previous
    #         y0 = y0 + y_i
    
    #     # delete first column (represents hisco=-2 - non-labelled case)
    #     # y0 = np.delete(y0, (0), 1)
    #     return(y0)
    
    # labels = y_to_mat(df, num_classes=key.shape[0])
    
    # labels = labels[:,2:]
    
    # %% Labels to list
    df_codes = df[["code1", "code2", "code3", "code4", "code5"]]
    
    # Load binarizer
    from sklearn.preprocessing import MultiLabelBinarizer
    one_hot_labels = MultiLabelBinarizer()
    
    # Binarize
    labels_list = df_codes.values.tolist()
    one_hot_labels = one_hot_labels.fit([key.code.values.tolist()])
    labels = one_hot_labels.transform(labels_list)
    labels = one_hot_labels.transform(labels_list)
    labels = labels[:, 3:] # Using only 3nd element and forward (the first 3 are various NA strings)
    labels = labels.astype(float)
    max_labels = labels.shape[1]
    
    # %% Eval metrics
    from keras import backend as K
    
    def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + 
        K.epsilon())
        return recall
    
    def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    
    def f1_m(y_true, y_pred):
        precision = precision_m(y_true, y_pred)
        recall = recall_m(y_true, y_pred)
        return 2*((precision*recall)/(precision+recall+K.epsilon()))
    
    # %% Define model
    y = labels
    
    num_rows, num_cols = y.shape
    x_string = df["occ1"]
    
    # Length of strings for max_length
    lengths = [len(s) for s in x_string]
    length99_999 = np.percentile(lengths, 99.999)
    length99_999 = int(length99_999)+1 # Ceiling
    
    # Define vectorization layer
    vectorize_layer = TextVectorization(
        max_tokens=1000,
        output_mode='int',
        output_sequence_length=length99_999,
        split = 'character'
        )
    
    # Adapt vectorizer
    vectorize_layer.adapt(x_string)
    
    # Define model
    mod_char = make_model1(
        n_outputs=y.shape[1],
        layer_size=512,
        max_length = length99_999,
        vectorize_layer = vectorize_layer
        )
    
    # %% Data generator
    def data_generator(X, y, batch_size=32, alt_prob = 0):
        """
        A generator that yields batches of training data.
        """
        num_samples = X.shape[0]
        indices = np.arange(num_samples)
        while True:
            # Shuffle indices
            np.random.shuffle(indices)
            for i in range(0, num_samples, batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_X = [X[j] for j in batch_indices]
                batch_X = Attacker(batch_X, alt_prob = alt_prob)
                batch_y = y[batch_indices]
                yield np.array(batch_X), np.array(batch_y)
                          
    
    # Reset indicies
    x_string = x_string.reset_index(drop=True)
                
    # Split data into training and validation sets
    split_index = int(0.9 * len(x_string))
    train_X = x_string[:split_index]
    train_y = y[:split_index]
    val_X = x_string[split_index:]
    val_y = y[split_index:]
    
    # Create generators for training and validation sets
    # train_generator = data_generator(train_X, train_y, batch_size=batch_size, alt_prob=0.05)
    # val_generator = data_generator(val_X, val_y, batch_size=32, alt_prob=0)
    
    # %% Train base models
    # Train in two steps. First without alterations. Then with alterations
    mod_char.fit(
        data_generator(train_X, train_y, batch_size=batch_size, alt_prob=alt_prob), 
        validation_data=[val_X, val_y],
        epochs=1, 
        steps_per_epoch=len(x_string)//batch_size,
        callbacks=[plt_callback1, callback2]
        )
    
    mod_char.fit(
        data_generator(train_X, train_y, batch_size=batch_size, alt_prob=alt_prob), 
        validation_data=[val_X, val_y],
        epochs=n_epochs, 
        steps_per_epoch=len(x_string)//batch_size,
        callbacks=[plt_callback1, callback2]
        )
    
    
    eval_res = mod_char.evaluate(x = val_X, y = val_y)
    pd.DataFrame([mod_char.metrics_names, eval_res]).to_csv('Data/Performance_char_aug'+str(alt_prob)+"_unique_"+str(unique_strings)+"_smplsize_"+str(sample_size)+"_lvl"+str(hisco_level)+".csv")
    
    # Loss to beat: 
    # val_loss: 1.2026e-04 - val_accuracy: 0.9642 - val_f1_m: 0.9719 - 
    # val_precision_m: 0.9750 - val_recall_m: 0.9702
    
    # %% Save models
    fname = 'Models/Mod_char_aug'+str(alt_prob)+"_unique_"+str(unique_strings)+"_smplsize_"+str(sample_size)+"_lvl"+str(hisco_level)
    mod_char.save(fname)
    
    # %% Load models
    # fname = "Models/Mod_char_aug0.2_unique_False_smplsize_10_lvl5"
    
    mod_char = tf.keras.models.load_model(fname, custom_objects={'f1_m':f1_m, 'precision_m': precision_m, 'recall_m': recall_m})
    
    # %% String to hisco
    def string_to_hisco(string, mod):
        pred = mod.predict([string])
        adress = np.where(pred>0.5)[1]+3
        prob = pred[0][adress-3]
        adress = list(adress)
        hisco = key.hisco[adress].tolist()
        desc = key.en_hisco_text[adress].tolist()
        res = pd.DataFrame({
        'string': [string] * len(adress),
        'hisco': hisco,
        'description': desc,
        'prob': prob
         })
        print(res)
        
    string_to_hisco("filosofisk uddannet", mod_char)
      

