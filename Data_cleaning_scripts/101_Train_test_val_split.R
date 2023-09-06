# Train test and validation split
# Updated:    2023-09-06
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

# Danish data
DK_census = loadRData("Data/Tmp_data/Clean_DK_census.Rdata")
DK_cedar  = loadRData("Data/Tmp_data/Clean_DK_cedar_translation.Rdata")
DK_orsted = loadRData("Data/Tmp_data/Clean_DK_orsted.Rdata")

# English data
EN_marr   = loadRData("Data/Tmp_data/Clean_EN_marr_cert.Rdata")
EN_parish = loadRData("Data/Tmp_data/Clean_EN_parish_records.Rdata")

# Dutch data
HSN_data  = loadRData("Data/Tmp_data/Clean_HSN_database.Rdata") # Dutch data

# Swedish data
SE_chalmers = loadRData("Data/Tmp_data/Clean_SE_chalmers.Rdata")

# ==== Train test val split ====
# This following is done to make sure that train/test/val split is entirely reproducible
# Generate common long vector of samples train, val, test
# space = c(rep("Train",7), rep("Val1", 1), rep("Val2", 1), rep("Test",1))
# set.seed(20)
# train_test_split = sample(space, 10^7, replace = TRUE)
# save(train_test_split, file = "Data/Manual_data/Random_sequence.Rdata")
load("Data/Manual_data/Random_sequence.Rdata")

# ==== Add split and clean data ====

# Danish data
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
DK_cedar = DK_cedar %>% 
  sample_frac(1) %>% 
  mutate(split = train_test_split[1:n()]) %>% 
  Validate_split() %>% 
  Keep_only_relevant()

set.seed(20)
DK_orsted = DK_orsted %>% 
  sample_frac(1) %>% 
  mutate(split = train_test_split[1:n()]) %>% 
  Validate_split() %>% 
  Keep_only_relevant()

# English data
set.seed(20)
EN_marr = EN_marr %>% 
  # Reshuffle
  sample_frac(1) %>% 
  mutate(split = train_test_split[1:n()]) %>% 
  Validate_split() %>% 
  Keep_only_relevant()

set.seed(20)
EN_parish = EN_parish %>% 
  # Reshuffle
  sample_frac(1) %>% 
  mutate(split = train_test_split[1:n()]) %>% 
  Validate_split() %>% 
  Keep_only_relevant()

# Dutch data
set.seed(20)
HSN_data = HSN_data %>% 
  # Reshuffle
  sample_frac(1) %>% 
  mutate(split = train_test_split[1:n()]) %>% 
  Validate_split() %>% 
  Keep_only_relevant()

# Swedish data
set.seed(20)
SE_chalmers = SE_chalmers %>% 
  sample_frac(1) %>% 
  mutate(split = train_test_split[1:n()]) %>% 
  Validate_split() %>% 
  Keep_only_relevant()

# ==== Save data ====
# Danish data
DK_census %>% 
  Save_train_val_test("DK_census", "da")

DK_cedar %>% 
  Save_train_val_test("DK_cedar", "da")

DK_orsted %>% 
  Save_train_val_test("DK_orsted", "da")

# English data
EN_marr %>% 
  Save_train_val_test("EN_marr_cert", "en")
EN_parish %>% 
  Save_train_val_test("EN_parish", "en")

# Dutch data
HSN_data %>% 
  Save_train_val_test("HSN_database", "nl")

# Swedish data
SE_chalmers %>% 
  Save_train_val_test("SE_chalmers", "se")

# ==== Training data stats ====
# Total training data
total = lapply(
  list(DK_census, DK_cedar, DK_orsted, EN_marr, HSN_data, SE_chalmers),
  function(x){NROW(x)}
) %>% 
  unlist() %>% 
  sum()

cat("\nTotal training data", total/10^6, "mil. observations")
  