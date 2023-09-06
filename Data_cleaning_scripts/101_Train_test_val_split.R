# Train test and validation split
# Created:    2023-05-24
# Auhtors:    Christian Vedel [christian-vs@sam.sdu.dk],
#
# Purpose:    Splits all clean tmp files into training, test and validation
#             Also converts HISCO codes into codes in the range 1:k, where k
#             is the maximum number of HISCO codes.
#
# Output:     Train, test and validation data

# ==== Libraries ====
library(tidyverse)
source("Data_cleaning_scripts/000_Functions.R")

# ==== Load data ====
DK_census = loadRData("Data/Tmp_data/Clean_DK_census.Rdata")
EN_marr   = loadRData("Data/Tmp_data/Clean_EN_marr_cert.Rdata")
HSN_data  = loadRData("Data/Tmp_data/Clean_HSN_database.Rdata") # Dutch data

# ==== Train test val split ====
# This following is done to make sure that train/test/val split is entirely reproducible
# Generate common long vector of samples train, val, test
# space = c(rep("Train",7), rep("Val1", 1), rep("Val2", 1), rep("Test",1))
# set.seed(20)
# train_test_split = sample(space, 10^7, replace = TRUE)
# save(train_test_split, file = "Data/Manual_data/Random_sequence.Rdata")
load("Data/Manual_data/Random_sequence.Rdata")

# ==== Add split and clean data ====
set.seed(20)
DK_census = DK_census %>% 
  # Delete unused vars:
  select(-Household_status, -Occupation, -labelled) %>% 
  # Reshuffle
  sample_frac(1) %>% 
  mutate(split = train_test_split[1:n()]) %>% 
  Validate_split() %>% 
  Keep_only_relevant()


set.seed(20)
EN_marr = EN_marr %>% 
  # Reshuffle
  sample_frac(1) %>% 
  mutate(split = train_test_split[1:n()]) %>% 
  Validate_split() %>% 
  Keep_only_relevant()

set.seed(20)
HSN_data = HSN_data %>% 
  # Reshuffle
  sample_frac(1) %>% 
  mutate(split = train_test_split[1:n()]) %>% 
  Validate_split() %>% 
  Keep_only_relevant()


# ==== Save data ====
DK_census %>% 
  Save_train_val_test("DK_census", "da")

EN_marr %>% 
  Save_train_val_test("EN_marr_cert", "en")

HSN_data %>% 
  Save_train_val_test("HSN_database", "nl")
