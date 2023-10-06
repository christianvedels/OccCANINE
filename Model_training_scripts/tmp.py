# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 10:25:32 2023

@author: chris
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score

from transformers import pipeline

import torch


classifier = pipeline("sentiment-analysis")
type(classifier)

# Output is a dictionary containing label and score as keys
classifier = pipeline("zero-shot-classification", device = 0)

text = "I want to book a flight from London to Paris."
labels = ["travel", "business", "food", "technology"]
classifier(text, labels)


text = [
    "He fishes but also has some land that he tends to",
    "He is a maker of fine dresses for the women of the town",
    "Retired farmer",
    "Fireman"
    ]

labels = ["fisherman", "farmer", "taylor", "works with trains", "not employed"]
classifier(text, labels)