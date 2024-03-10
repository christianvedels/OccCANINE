Breaking the HISCO Barrier: Automatic Occupational Standardization with OccCANINE
=====================
*Christian Møller Dahl & Christian Vedel, University of Southern Denmark*

--------
Welcome to the GitHub repository for OccCANINE, a tool designed to transform occupational descriptions into standardized HISCO (Historical International Standard Classification of Occupations) codes automatically. Developed by Christian Møller Dahl and Christian Vedel from the University of Southern Denmark, this tool leverages the power of a finetuned language model to process and classify occupational descriptions with high accuracy, precision, and recall. 

Paper: [https://arxiv.org/abs/2402.13604](https://arxiv.org/abs/2402.13604)

Huggingface: [Christianvel/OccCANINE](https://huggingface.co/Christianvedel/OccCANINE)

Slides: [Breaking the HISCO barrier](https://raw.githack.com/christianvedels/OccCANINE/main/Project_dissemination/HISCO%20Slides/Slides.html)


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

OccCANINE folder
-------------
The OccCANINE folder contains all the code used for training and application of OccCANINE. 

*   **Data/**: Contains 'key.csv' which maps integer codes (generated by OccCANINE) to HISCO codes. It also contains toydata to use when trying out OccCANINE for the first time.
*   **Finetuned/**: Contains any locally finetuned models.
*   **Model/**: Contains the OccCANINE model binary
*   **n001_Model_assets.py**: Defines the unlerying pytorch model
*   **n100_Attacker.py**: Defines text attack procedure used for text augmentation in training.
*   **n101_Trainer.py**: Defines training procedures.
*   **n102_DataLoader.py**: Defines how data is loaded and fed to the model in training.
*   **n103_Prediction_assets.py**: Functions and classes to use OccCANINE. This also contains the 'OccCANINE' class.
*   **n201_TrainCANINE.py**: This is the script was used to train OccCANINE.
*   **n302_Add_PyTorchModelHubMixin_to_model**: Is a small script, which was used to make OccCANINE readable from Hugging Face

Model_evaluation_scripts folder
-------------
The Model_evaluation_scripts folder contains all the code used to generate the model evaluation results shown in the paper. 

### Python scripts
*   **n001_Predict_eval.py**: Runs predictions on 1 million validation observations.
*   **n002_Copenhagen_burial.py**: Runs predictions on 200 observations from the Copenhagen Burial Records from [Link Lives](https://www.rigsarkivet.dk/udforsk/link-lives-data/)
*   **n003_Training_ship_data.py**: Runs predictions on 200 observations of parent's occupations from the [Indefatigable training ship](https://reshare.ukdataservice.ac.uk/853251/)
*   **n004_Dutch_familiegeld.py**: Runs predictions on 200 observations of occupations in the Dutch familiegeld 
*   **n005_Swedish_strikes.py**: Runs predictions on 200 observations of the profession of [Swedish strikes]([https://reshare.ukdataservice.ac.uk/853251/](https://hdl.handle.net/10622/TAVJXR))

### R scripts
*   **000_Functions.R**: Contains functions used in evaluation.
*   **001_Generate_eval_stats.R**: Generates accuracy, precision, etc. for validation data across various subgroups.
*   **002_Nature_of_mistakes.R**: Returns plots and statistics which generate insights into the nature of mistakes, when OccCANINE disagrees with the validaiton data.
*   **101_Eval_illustrations.R**: Generates most of the illustrations and statistics shown in the paper.
*   **102_Embeddings_visualisation.R**: This makes the embedding t-sne illustrations.

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

