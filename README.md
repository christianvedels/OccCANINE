# Automatic HISCO codes

## Structure
- 'Data_cleaning_scripts' contains R scripts which takes data from 'Data/Raw_data' into a format suitable for training (which is then stored in 'Data/Training_data', 'Data/Validation_data' and 'Data/Test_data')
- 'Model_training_scripts' contains Python scripts which trains models and stores them in 'Trained_models' 
- 'PREDICT_HISCOs.py' is the main product. This script takes occupational descriptions and outputs HISCO codes for them. 

### Large files
'Raw_data', 'Test_data' and 'Training_data' are each too large to redistribute via github. Each of these directories contains a text-file 'DROPBOX.txt' with a dropbox link to where the file can be downloaded.

## Data cleaning
Data cleaning scripts are found in 'Data_cleaning_scripts'. Everything runs in numeric order the scripts name.

- '000_Function.R' contains functions shared across all data cleaning.  
- '001_Assets_for_cleaning.R' generates assets for data cleaning to avoid miscelaneous difference in key features. E.g. the encoding of hisco to 1 to N encoding.  
- '00[x]_...R' when x>1 cleans individual data sources. Each of these save one file 'Clean_....Rdata' containing cleaned data to use in the final following step.  
- '101_Train_test_val_split.R' makes everything consistent and save training, validation and test data

The procedure for generating traning can be outlined as follows: 
1. **Sanity check:** Data is loaded and manually checked for content, consistency of labels, etc. 
2. **Extracting relevant data:**  Relevant variables are extracted. In some cases the raw data contains both a 'raw' and a 'cleaned' occupational description. In these cases each are kept as separate observations. 
3. **Combinations:** We want to be able to identify cases with more than one occupation. This naturally occures in many of the data sources. But data sources which are merely lists occupational descriptions and their hisco code (see below), this does not occur by construction.  We construct this synethetically by combining occupational descriptions with the relevant 'and' in that language. E.g. "Occupied as a fisherman": 64100 and "Tends to some land": 61110 becomes "Occupied as a fisherman and tends to some land": (64100, 61110). 
4. **Filtering:** To make sure that all HISCO codes are valid we filter off any observations for which HISCO codes are not found on the list provided by the ['hisco' R library](https://github.com/cedarfoundation/hisco)  

Data types:
- Census
- Marriage records
- Descriptions 


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
| Source file name    | Observations  | Description                                              | Reference   | Language |
|----------------------|---------------|---------------------------------------------------------|-------------|----------|
| EN_uk_ipums_[x].csv  | 5,108,577     | English UK IPUMS dataset                                | MPC (2020)     | en |
| DK_census_[x].csv    | 4,673,892     | Danish census data.                                     | Clausen (2015) | da |
| EN_marr_cert_[x].csv | 4,046,387     | English marriage certificate data                       | Unknown     | en |
| SE_swedpop_[x].csv   | 1,793,557     | Swedish population data                                 | Unknown     | se |
| EN_ca_ipums_[x].csv  | 1,427,559     | English Canada IPUMS dataset                            | Unknown     | en |
| JIW_database_[x].csv | 966,793       | JIW database                                            | Unknown     | nl |
| CA_bcn_[x].csv       | 644,484       | Canadian Barcelona dataset                              | Unknown     | ca |
| EN_us_ipums_[x].csv  | 313,069       | English US IPUMS dataset                                | Unknown     | en |
| NO_ipums_[x].csv     | 233,518       | Norwegian IPUMS dataset                                 | Unknown     | no |
| FR_desc_[x].csv      | 142,778       | French description data                                 | Unknown     | fr |
| HSN_database_[x].csv | 134,957       | HSN database                                            | Unknown     | nl |
| EN_parish_[x].csv    | 73,806        | English parish dataset                                  | Unknown     | en |
| IS_ipums_[x].csv     | 47,497        | Icelandic IPUMS dataset                                 | Unknown     | is |
| DK_cedar_[x].csv     | 46,563        | Danish cedar dataset                                    | Unknown     | da |
| SE_cedar_[x].csv     | 45,581        | Swedish cedar dataset                                   | Unknown     | se |
| DE_ipums_[x].csv     | 36,942        | German IPUMS dataset                                   | Unknown     | de |
| DK_orsted_[x].csv    | 36,608        | Danish Orsted dataset                                  | Unknown     | da |
| EN_oclack_[x].csv    | 24,530        | English Oclack dataset                                  | Unknown     | en |
| EN_loc_[x].csv       | 23,179        | English Loc dataset                                    | Unknown     | en |
| SE_chalmers_[x].csv  | 14,426        | Swedish Chalmers dataset                               | Unknown     | se |
| IT_fm_[x].csv        | 4,525         | Italian FM dataset                                     | Unknown     | it |


x $\in$ {train, test, val}


## Updates

### 2023-11-06
- Added evaluation scripts
- Added several more languages 
- Added retrain scripts
- Added embedding visualisation 

### 2023-10-11
- Started writing evaluation scripts. They don't work yet.

### 2023-10-08
- XML-RoBERTa works. The entry point is 'n202_TrainXML_RoBERTa.py'

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

## References
Clausen, N. F. (2015). The Danish Demographic Database—Principles and Methods for Cleaning and Standardisation
of Data. Population Reconstruction, 1–22. https://doi.org/10.1007/978-3-319-19884-2_1

Clark, Gregory and Cummins, Neil and Curtis, Matthew (2022) The Mismeasure of Man: Why Intergenerational Occupational Mobility is Much Lower than Conventionally Measured, England, 1800-2021. CEPR Discussion Paper No. DP17346, Available at SSRN: https://ssrn.com/abstract=4144664

Minnesota Population Center *MPC* (2020) Integrated Public Use Microdata Series, International: Version 7.3 [dataset]. Minneapolis, MN: IPUMS, 2020. https://doi.org/10.18128/D020.V7.3

SwedPop. (2022). Version 2 of harmonised data in SwedPop. SwedPop. https://doi.org/10.48524/SWEDPOP-V2-0

De Moor, Tine; van Weeren, René (2021). Dataset Ja, ik wil - Amsterdam marriage banns registers 1580-1810. Erasmus University Rotterdam (EUR). Dataset. https://doi.org/10.25397/eur.14049842.v1

Pujades Mora, Joana Maria; Valls, Miquel (2017) "Barcelona Historical Marriage Database", https://hdl.handle.net/10622/SDZPFE, IISH Data Collection, V1, UNF:6:f3xC63ywNbwbs3jyZbQhMQ== [fileUNF]

Mandemakers, Kees; Hornix, Jan; Mourits, Rick; Muurling, Sanne; Boter, Corinne; Van Dijk, Ingrid K; Maas, Ineke; Van de Putte, Bart; Zijdeman, Richard L; Lambert, Paul; Van Leeuwen, Marco H D; Van Poppel, Frans W A; Miles, Andrew (2020), "HSNDB Occupations", https://hdl.handle.net/10622/88ZXD8, IISH Data Collection, V1; HSN_HISCO_release_2020_02.tab [fileName], UNF:6:PcnlkBzFeCgwnL0sI5KQGw== [fileUNF]

de Pleijt, A. Nuvolari, A. Weisdorf, J. (2020) Human Capital Formation During the First Industrial Revolution: Evidence from the use of Steam Engines, Journal of the European Economic Association (18) 829–889, https://doi.org/10.1093/jeea/jvz006

Mooney, G. (2016) "Mooney_1866_London_occupational_codes", https://hdl.handle.net/10622/ERGY0V, IISH Data Collection, V1; lndn1866_01.tab [fileName], UNF:6:5FaJFA/ckntMZWpx6jtOtA== [fileUNF]

A. Fornasin; A. Marzona (2016) "HISCO_Italian_Formasin_Marzona_2006", https://hdl.handle.net/10622/SRVW6S, IISH Data Collection, V1; HISCO_Italian_Formasin_Marzona_2006.tab [fileName], UNF:6:fUt1c768wtTbCN7L0OspLA== [fileUNF]
