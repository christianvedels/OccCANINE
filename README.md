Breaking the HISCO Barrier: Automatic Occupational Standardization with OccCANINE
=====================
Welcome to the GitHub repository for OccCANINE, a tool designed to transform occupational descriptions into standardized HISCO (Historical International Standard Classification of Occupations) codes automatically. Developed by Christian Møller Dahl and Christian Vedel from the University of Southern Denmark, this tool leverages the power of a finetuned language model to process and classify occupational descriptions with high accuracy, precision, and recall. 

Paper: **LINK**

Huggingface: **LINK**

Getting started
--------
- See the [colab notebook](https://github.com/christianvedels/OccCANINE/blob/main/OccCANINE_colab.ipynb) for a demonstration of OccCANINE
- To use the model at scale please clone/download the repository. Open [PREDICT_HISCOs.py](https://github.com/christianvedels/OccCANINE/blob/main/PREDICT_HISCOs.py) to get started.
- A step-by-step installation guide can be found in [GETTING_STARTED.md](https://github.com/christianvedels/OccCANINE/blob/main/GETTING_STARTED.md)

Overview
--------

This repository provides everything needed to generate automatic HISCO codes from occupational descriptions using OccCANINE. It also provides replication files for all steps from raw training data to the final trained. 

Structure
---------

*   **Data\_cleaning\_scripts**: Contains R scripts for processing raw data from 'Data/Raw\_data' into a format suitable for training, which is then stored in 'Data/Training\_data', 'Data/Validation\_data', and 'Data/Test\_data'.
*   **OccCANINE**: Contains Python scripts for training OccCANINE and using the already finetuned version of it.
*   **PREDICT\_HISCOs.py**: The main product that takes occupational descriptions and outputs HISCO codes.

Data Cleaning
-------------

Scripts for data cleaning are located in 'Data\_cleaning\_scripts' and should be run in numeric order as indicated by the script names.

*   **000\_Function.R**: Contains functions shared across all data cleaning scripts.
*   **001\_Assets\_for\_cleaning.R**: Generates assets for data cleaning, such as the encoding of HISCO to a 1 to N encoding.
*   **00\[x\]\_...R** (where x>1): Cleans individual data sources, saving one file 'Clean\_....Rdata' for each source.
*   **101\_Train\_test\_val\_split.R**: Ensures consistency and saves training, validation, and test data.

### Data Cleaning Process

1.  **Sanity Check**: Manual verification of data content and consistency.
2.  **Extracting Relevant Data**: Extraction of relevant variables, keeping both 'raw' and 'cleaned' occupational descriptions as separate observations when available.
3.  **Combinations**: Synthetic creation of descriptions representing more than one occupation by combining descriptions with the respective language's word for 'and'.
4.  **Filtering**: Removal of observations with invalid HISCO codes based on the ['hisco' R library](https://github.com/cedarfoundation/hisco).

### Structure of Training Data

The training data is structured with variables including the year of observation, a unique ID for every observation, the occupational description string, HISCO codes, integer codes for HISCO, the language of the description, and a string indicating the data split (train, val1, val2, or test).

