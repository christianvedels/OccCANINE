# Train test and validation split
# Updated:    2025-10-25
# Auhtors:    Christian Vedel [christian-vs@sam.sdu.dk],
#
# Purpose:    Creates a unique string version of the test data
#
# Output:     Unique string test data

# ==== Libraries ====
library(tidyverse)
library(foreach)
source("Data_cleaning_scripts/000_Functions.R")

# ==== Create unique training data =====
# A problem is that some strings might appear (by chance) in both train, val and test
files_train = list.files("Data/Training_data", full.names = TRUE)
files_test = list.files("Data/Test_data", full.names = TRUE)

unique_train_strings = foreach(f = files_train, .combine = "bind_rows") %do% {
  data = read_csv(f, col_types = "cccccccccccccccccccccccccccccccccccccc", progress = FALSE)
  cat("\nProcessing", f)
  train_strings = data %>%
    filter(split == "Train") %>%
    distinct(occ1, lang)
  
  return(train_strings)
}

unique_test_strings = foreach(f = files_test, .combine = "bind_rows") %do% {
  data = read_csv(f, col_types = "cccccccccccccccccccccccccccccccccccccc", progress = FALSE)
  cat("\nProcessing", f)
  test_strings = data %>%
    filter(split %in% c("Test")) %>%
    distinct(occ1, lang)

  return(test_strings)
}

# Test not in train
unique_test_only = unique_test_strings %>%
  anti_join(unique_train_strings, by = c("occ1", "lang"))

rm(unique_train_strings, unique_test_strings)

# Create dir if not exists
if(!dir.exists("Data/Test_data_unique_strings")){
  dir.create("Data/Test_data_unique_strings")
}

# Load test data and filter
n_original = 0
n_unique = 0
foreach(f = files_test, .combine = "bind_rows") %do% {
  data = read_csv(f, col_types = "cccccccccccccccccccccccccccccccccccccc", progress = FALSE)
  n_original = n_original + nrow(data)
  cat("\nProcessing", f)
  test_data_unique = data %>%
    semi_join(unique_test_only, by = c("occ1", "lang"))

  n_unique = n_unique + nrow(test_data_unique)
  cat("\nReduced from", nrow(data), "to", nrow(test_data_unique), "rows")
  # Save
  name = str_remove(basename(f), ".csv")
  write_csv(test_data_unique, paste0("Data/Test_data_unique_strings/", name, "_unique_strings.csv"))
}

cat("\nOriginal test data rows:", n_original)
cat("\nUnique string test data rows:", n_unique)
cat("\nReduction:", round(100*(n_original - n_unique)/n_original,2), "%")



