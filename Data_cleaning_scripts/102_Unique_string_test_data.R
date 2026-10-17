# Train test and validation split
# Updated:    2026-10-17
# Auhtors:    Christian Vedel [christian-vs@sam.sdu.dk],
#
# Purpose:    Creates a unique string version of the test data, i.e. test
#             observations whose string (and language) is not found in the
#             training data of the same set of sources.
#
# Output:     Unique string test data
#             - Data/Test_data_unique_strings:       All sources
#             - Data/Test_data_unique_strings_paper: Only sources used in the paper
#               (the paper's model was not trained on sources added later)

# ==== Libraries ====
library(tidyverse)
library(foreach)
source("Data_cleaning_scripts/000_Functions.R")

# Sources added after the paper's model was trained
SOURCES_NOT_IN_PAPER = c("Mosaic_data", "Leeuwen_data", "Handheld_data")

# ==== make_unique_string_test ====
# A problem is that some strings might appear (by chance) in both train, val and test
# exclude_sources: Sources left out of both training and test data
# dir_out:         Output directory
make_unique_string_test = function(exclude_sources = NULL, dir_out){
  read0 = function(f) read_csv(f, col_types = cols(.default = "c"), progress = FALSE, trim_ws = FALSE)
  keep_file = function(f) !str_remove(basename(f), "_(train|test)\\.csv$") %in% exclude_sources

  files_train = list.files("Data/Training_data", full.names = TRUE)
  files_test = list.files("Data/Test_data", full.names = TRUE)
  files_train = files_train[sapply(files_train, keep_file)]
  files_test = files_test[sapply(files_test, keep_file)]

  unique_train_strings = foreach(f = files_train, .combine = "bind_rows") %do% {
    cat("\nProcessing", f)
    read0(f) %>%
      filter(split == "Train") %>%
      distinct(occ1, lang)
  }

  unique_test_strings = foreach(f = files_test, .combine = "bind_rows") %do% {
    cat("\nProcessing", f)
    read0(f) %>%
      filter(split %in% c("Test")) %>%
      distinct(occ1, lang)
  }

  # Test not in train
  unique_test_only = unique_test_strings %>%
    anti_join(unique_train_strings, by = c("occ1", "lang"))

  rm(unique_train_strings, unique_test_strings)

  # Create dir if not exists
  if(!dir.exists(dir_out)){
    dir.create(dir_out)
  }

  # Load test data and filter
  n_original = 0
  n_unique = 0
  for(f in files_test){
    data = read0(f)
    n_original = n_original + nrow(data)
    cat("\nProcessing", f)
    test_data_unique = data %>%
      semi_join(unique_test_only, by = c("occ1", "lang"))

    n_unique = n_unique + nrow(test_data_unique)
    cat("\nReduced from", nrow(data), "to", nrow(test_data_unique), "rows")
    # Save
    name = str_remove(basename(f), ".csv")
    write_csv(test_data_unique, paste0(dir_out, "/", name, "_unique_strings.csv"))
  }

  cat("\n\n", dir_out)
  cat("\nOriginal test data rows:", n_original)
  cat("\nUnique string test data rows:", n_unique)
  cat("\nShare kept:", round(100*(n_unique)/n_original,2), "%\n")
}

# ==== Run ====
make_unique_string_test(dir_out = "Data/Test_data_unique_strings")
make_unique_string_test(exclude_sources = SOURCES_NOT_IN_PAPER, dir_out = "Data/Test_data_unique_strings_paper")
