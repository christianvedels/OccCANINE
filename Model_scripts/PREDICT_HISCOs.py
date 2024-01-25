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
    )

# %% Example 1
model.predict(
    ["taylor of fne dresses and mechanic of motor vehicles"],
    what = "pred"
    )

# %% Example 2
model.predict(
    ["linotype operator of the press"],
    what = "pred"
    )

# %% Example 3 (language specified)
model.predict(
    ["Detta är bagaren som bakar de bästa kardemummabullarna i stan"],
    lang="se",
    what = "pred"
    )


# %%
model.predict(
    ["bakar"],
    lang="se",
    what = "pred"
    )

# %% Example 3 (without language)
model.predict(
    ["Detta är bagaren som bakar de bästa kardemummabullarna i stan"],
    what = "pred"
    )
