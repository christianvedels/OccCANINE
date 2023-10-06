# Automatic HISCO codes

## Structure
- 'Data_cleaning_scripts' contains R scripts which takes data from 'Data/Raw_data' into a format suitable for training (which is then stored in 'Data/Training_data', 'Data/Validation_data' and 'Data/Test_data')
- 'Model_training_scripts' contains Python scripts which trains models and stores them in 'Trained_models' 
- 'PREDICT_HISCOs.py' is the main product. This script takes occupational descriptions and outputs HISCO codes for them. 

### Large files
'Raw_data', 'Test_data' and 'Training_data' are each too large to redistribute via github. Each of these directories contains a text-file 'DROPBOX.txt' with a dropbox link to where the file can be downloaded.


### Structure of training data
| Variable              | Description                                                                                              |
|-----------------------|----------------------------------------------------------------------------------------------------------|
| Year | Year of observaiton, if any |
| RowID | Unique ID for every observation |
|occ1 | Occupational description string |
|hisco_[x] | All hisco codes, which apply - up to 5 (x $\in$ 1:5) |
|code_[x] | Hisco code translated to integer codes 0:1920 |
|lang | What language is it? |
|split | String describing if this is 'train', 'val1', 'val2' or 'test' |

## Overview of training data and sources
| Source file name | Observations | Description          | Reference    |
|------------------|--------------|----------------------|--------------|
| DK_census_[a].csv| 4,673,892    | Danish census data from the Danish Demographic Database. This also contains HISCO codes for some census years | Clausen (2015) |

a $\in$ {train, test, val}


## Updates

### 2023-10-06
- Added XML-RoBERTa with language strings in 0.75 of cases 
- Updated the rest of model training scripts accordingly

### 2023-09-06
- Added 'language' as a column in all training data

### 2023-08-24
- IPUMS has their own adapted HISCO system. Added a 'key_ipums.csv' based on this

### 2023-08-17
- Refactored all model training code entrance point is 'n201_TrainBERT.py'

### 2023-08-13
- English marriage certificate data ready for training
- 004_Danish_BERT.py now runs. But the code is a mess

### 2023-07-06
- Added proof of concept XML-RoBERTa model

### 2023-07-01
- DK training data ready

### 2023-05-30
- Added train/test/val split to DK census data
- Added NA-padding to assets and DK census cleaning such that NA and no occupation are treated differently

### 2023-05-29
- Added manually verified observations with no occupation from the Danish census data
- Finished Danish census data cleaning script

### 2023-05-23 
- Copy pasted content from old version of the project
- Added blank data cleaning scripts
- Added blank model training scripts

## Data sources
#### HISCO codes for Danish census data
Clausen, N. F. (2015). The Danish Demographic Database—Principles and Methods for Cleaning and Standardisation
of Data. Population Reconstruction, 1–22. https://doi.org/10.1007/978-3-319-19884-2_1

#### HISCO codes for English marrriage certificates
Clark, Gregory and Cummins, Neil and Curtis, Matthew (2022) The Mismeasure of Man: Why Intergenerational Occupational Mobility is Much Lower than Conventionally Measured, England, 1800-2021. CEPR Discussion Paper No. DP17346, Available at SSRN: https://ssrn.com/abstract=4144664

#### HISCO codes from IPUMS International
Minnesota Population Center (2020) Integrated Public Use Microdata Series, International: Version 7.3 [dataset]. Minneapolis, MN: IPUMS, 2020. https://doi.org/10.18128/D020.V7.3
