# Train test and validation split - other systems of encoding
# Updated:    2024-08-21
# Auhtors:    Christian Vedel [christian-vs@sam.sdu.dk],
#
# Purpose:    Splits all clean tmp files of other systems of encoding into
#             Train/test/val split.
#
# Output:     Train, test and validation data

# ==== Libraries ====
library(tidyverse)
source("Data_cleaning_scripts/000_Functions.R")
library(foreach)

# ==== Pipeline ====
pipeline = function(x, name, lang, samples = FALSE){
  # Check if name already exists
  test = any(grepl(name, list.files("Data/Training_data_other")))
  if(test){
    cat("\n",name,"Data already processed")
    return(name)
  }
  
  cat("\nLoading", x)
  x = loadRData(x)
  
  if(name %in% c("EN_uk_ipums", "EN_us_ipums")){
    # These are downsampled to not dominate training
    x = x %>% 
      select(-RowID) %>% 
      distinct() %>% 
      mutate(RowID = 1:n())
  }
  
  load("Data/Manual_data/Random_sequence_long.Rdata")
  set.seed(20)
  
  cat("\nMaking data ready")
  x = x %>%
    # Reshuffle
    sample_frac(1) %>%
    mutate(split = train_test_split[1:n()]) %>%
    Validate_split() %>%
    Keep_only_relevant(
      other_to_keep = c(
        "synthetic_combination", 
        paste0("PSTI_", 1:5),
        paste0("OCC1950_", 1:5),
        paste0("OCCICEM_", 1:5),
        paste0("ISCO68A_", 1:5)
      )
    )
  
  cat(paste("\nVars:", paste(names(x), collapse = ", ")))
  
  cat("\nSaving data")
  x %>% 
    Save_train_val_test(name, lang, dir = "other")
  cat("\nSaved:", name, lang)
  
  # Produce smaller samples (only train)
  if(samples){
    options(scipen=999)
    # Sample sizes
    samples_sizes = 10^(3:5)
    
    # Data prep
    x_relevant = x %>% filter(split == "Train")
    
    x_unique = x_relevant %>% 
      select(-RowID) %>% 
      distinct() %>% 
      mutate(
        RowID = 1:n()
      )
    
    # Produce samples of N observations
    set.seed(20)
    for(s in samples_sizes){
      x_relevant %>% 
        slice_sample(n = s) %>% 
        Save_train_val_test(paste0(name,"_n",s), lang, dir = "other")
      cat("\nSaved:", name, lang)
      
    }
    
    # Produce samples with unique strings
    set.seed(20)
    for(s in samples_sizes){
      x_unique %>% 
        slice_sample(n = s) %>% 
        Save_train_val_test(paste0(name,"_n_unq",s), lang, dir = "other")
      cat("\nSaved:", name, lang)
      
    }
  }
  
  return(name)
}

# ==== Run pipelines ====
lang = "en"
x = pipeline(
  "Data/Tmp_data/EN_PSTI.Rdata",
  "EN_PSTI_CAMPOP",
  lang,
  samples = TRUE
)

x = pipeline(
  "Data/Tmp_data/EN_IPUMS_OCC1950.Rdata",
  "EN_OCC1950_IPUMS_US",
  lang,
  samples = TRUE
)

x = pipeline(
  "Data/Tmp_data/EN_IPUMS_UK_OCCICEM.Rdata",
  "EN_OCCICEM_IPUMS_UK",
  lang,
  samples = TRUE
)

x = pipeline(
  "Data/Tmp_data/EN_IPUMS_UK_ISCO68.Rdata",
  "EN_ISCO68_IPUMS_UK",
  lang,
  samples = TRUE
)


# Check var names
train_fs = list.files("Data/Training_data_other", full.names = TRUE)
for(f in train_fs){
  read_csv(f, n_max = 100) %>% names() %>% print()
}


