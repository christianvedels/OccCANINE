# Automatic HISCO codes

## Structure
- 'Data_cleaning_scripts' contains R scripts which takes data from 'Data/Raw_data' into a format suitable for training (which is then stored in 'Data/Training_data', 'Data/Validation_data' and 'Data/Test_data')
- 'Model_training_scripts' contains Python scripts which trains models and stores them in 'Trained_models' 
- 'PREDICT_HISCOs.py' is the main product. This script takes occupational descriptions and outputs HISCO codes for them. 

### Large files
'Raw_data', 'Test_data' and 'Training_data' are each too large to redistribute via github. Each of these directories contains a text-file 'DROPBOX.txt' with a dropbox link to where the file can be downloaded.


## Updates

### 2023-05-23 
- Copy pasted content from old version of the project
- Added blank data cleaning scripts



## Data sources
#### HISCO codes for Danish census data
Clausen, N. F. (2015). The Danish Demographic Database—Principles and Methods for Cleaning and Standardisation
of Data. Population Reconstruction, 1–22. https://doi.org/10.1007/978-3-319-19884-2_1

#### HISCO codes for English marrriage certificates
Clark, Gregory and Cummins, Neil and Curtis, Matthew, The Mismeasure of Man: Why Intergenerational Occupational Mobility is Much Lower than Conventionally Measured, England, 1800-2021 (May 1, 2022). CEPR Discussion Paper No. DP17346, Available at SSRN: https://ssrn.com/abstract=4144664
