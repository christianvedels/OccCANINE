# -*- coding: utf-8 -*-
"""
Created on 2024-01-15

@author: christian-vs

SETUP:
    - See GETTING_STARTED.md (https://github.com/christianvedels/OccCANINE/blob/main/GETTING_STARTED.md)
"""

# Import necessary modules
from histocc import OccCANINE
import pandas as pd

# Load model
model = OccCANINE(
    verbose = True # To see updates while running prediction
    )

if __name__ == '__main__':
    # Load data
    df = pd.read_csv("histocc/Data/TOYDATA.csv")

    # Get HISCO codes
    result = model.predict(
        df["occ1"],
        lang = "en",
        threshold = 0.22 # Optimal for F1 in English (see the paper https://arxiv.org/abs/2402.13604)
        )
    
    # Save results
    result.to_csv("Tmp.csv")