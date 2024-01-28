# -*- coding: utf-8 -*-
"""
Created on 2024-01-15

@author: christian-vs

Prediction for applications

SETUP:
    - See readme2.md
"""

import os
script_directory = os.path.dirname(os.path.abspath(__name__))
os.chdir(script_directory)

# %% Import necessary modules
from n103_Prediction_assets import Finetuned_model

# %% Load model
model = Finetuned_model(
    name = "CANINE_Multilingual_CANINE_sample_size_10_lr_2e-05_batch_size_256",
    device = "cpu"
    )

# %% Texts
occupations = [
    "labourer and sailor",
    "production worker and unemployed",
    "stewardess",
    "clerk and labourer",
    "gasser and retired",
    "charwoman and tailor",
    "fireman in hospital",
    "hatter c and royal navy",
    "grocer and work study engineer",
    "formerly medical and medical doctor",
    "goods porter and unemployed",
    "labourer and carpenter",
    "gardener labourer and barber",
    "police constable and machine operator",
    "computor systems analyst and unemployed plasterer",
    "secretary and washing",
    "laundress and cleaner",
    "sailor and hotel proprietor",
    "labourer",
    "aircraft inspector and key smith",
    "post office clerk retired",
    "millenar at messr leas and district midwife",
    "shop assistant and keeps a lodging house",
    "machine operator and gardener",
    "shopkeeper and trade union official",
    "ma and carter",
    "civil service clerk and sailor",
    "ships engineer stepfather and divorced",
    "shop assistant and housewife"
]

model.predict(
    occupations,
    what = "pred"
    )


# %% Example 1
model.predict(
    ["taylor of fne dresses and mechanic of motor vehicles"],
    what = "pred"
    )


# %% Example 3 (language specified)
model.predict(
    ["Detta är bagaren som bakar de bästa kardemummabullarna i stan"],
    lang="se",
    what = "pred"
    )

# %% Example 3 (without language)
model.predict(
    ["Detta är bagaren som bakar de bästa kardemummabullarna i stan"],
    what = "pred"
    )
