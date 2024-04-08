Breaking the HISCO Barrier: Automatic Occupational Standardization with OccCANINE
=====================
*Christian Møller Dahl, Torben Johansen, Christian Vedel,*
*University of Southern Denmark*

--------
Welcome to the GitHub repository for OccCANINE, a tool designed to transform occupational descriptions into standardized HISCO (Historical International Standard Classification of Occupations) codes automatically. Developed by Christian Møller Dahl, Torben Johansen and Christian Vedel from the University of Southern Denmark, this tool leverages the power of a finetuned language model to process and classify occupational descriptions with high accuracy, precision, and recall. 

Paper: [https://arxiv.org/abs/2402.13604](https://arxiv.org/abs/2402.13604)

Huggingface: [Christianvel/OccCANINE](https://huggingface.co/Christianvedel/OccCANINE)

Slides: [Breaking the HISCO Barrier](https://raw.githack.com/christianvedels/OccCANINE/main/Project_dissemination/HISCO%20Slides/Slides.html)

How to use OccCANINE: [YouTube video](https://youtu.be/BF_oNe-sABQ?si=uEgNYLtPGNYAXCDK)

Getting started
--------
- See the [colab notebook](https://github.com/christianvedels/OccCANINE/blob/main/OccCANINE_colab.ipynb) for a demonstration of OccCANINE
- A step-by-step installation guide can be found in [GETTING_STARTED.md](https://github.com/christianvedels/OccCANINE/blob/main/GETTING_STARTED.md)
- Run `python predict.py --fn-in path/to/input/data.csv --col occ1 --fn-out path/to/output/data.csv --language [lang]` in the command line to get HISCO codes for all the descriptions found in the `occ1` column in the inputted data. See [predict.py](https://github.com/christianvedels/OccCANINE/blob/main/predict.py) for details.
- To see a simple script which reads data and uses OccCANINE to obtain HISCO codes see  [PREDICT_HISCOs.py](https://github.com/christianvedels/OccCANINE/blob/main/PREDICT_HISCOs.py).

Overview
--------

This repository provides everything needed to generate automatic HISCO codes from occupational descriptions using OccCANINE. It also provides replication files for all steps from raw training data to the final trained model. 

Structure
---------

*   **Data\_cleaning\_scripts**: Contains R scripts for processing raw data from 'Data/Raw\_data' into a format suitable for training, which is then stored in 'Data/Training\_data', 'Data/Validation\_data', and 'Data/Test\_data'.
*   **histocc**: Contains Python scripts for training OccCANINE and using the already finetuned version of it.
*   **Model_evaluation_scripts**: Contains a mix of R and Python scripts which generates model evaluation statistics and plots of these, which are found in the associated paper.

histocc folder
-------------
The histocc folder contains all the code used for training and application of OccCANINE. 

*   **Data/**: Contains 'key.csv' and which maps integer codes (generated by OccCANINE) to HISCO codes based off definitions by https://github.com/cedarfoundation/hisco. It also contains toydata to use when trying out OccCANINE for the first time.
*   **model_assets.py**: Defines the unlerying pytorch model
*   **attacker.py**: Defines text attack procedure used for text augmentation in training.
*   **trainer.py**: Defines training procedures.
*   **dataLoader.py**: Defines how data is loaded and fed to the model in training.
*   **prediction_assets.py**: Functions and classes to use OccCANINE. This also contains the 'OccCANINE' class, which serves as the main user interface in most cases.

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

