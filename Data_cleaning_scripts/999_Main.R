# MAIN
# Created:    2023-08-13
# Auhtors:    Christian Vedel [christian-vs@sam.sdu.dk],
#
# Purpose:    Runs all data cleaning scripts

source("Data_cleaning_scripts/000_Functions.R")
source("Data_cleaning_scripts/001_Assets_for_cleaning.R")
source("Data_cleaning_scripts/002_Cleaning_DK_census.R")
source("Data_cleaning_scripts/003_Cleaning_EN_marr_cert.R")
source("Data_cleaning_scripts/004_Cleaning_IPUMS.R")
source("Data_cleaning_scripts/101_Train_test_val_split.R")
