# Train test and validation split
# Updated:    2023-11-06
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
library(foreach)

# ==== Train test val split (run once ever) ====
# This following is done to make sure that train/test/val split is entirely reproducible
# Generate common long vector of samples train, val, test
# space = c(rep("Train",17), rep("Val1", 1), rep("Val2", 1), rep("Test",1))
# set.seed(20)
# train_test_split = sample(space, 10^7, replace = TRUE)
# save(train_test_split, file = "Data/Manual_data/Random_sequence.Rdata")
# load("Data/Manual_data/Random_sequence.Rdata")
# # Add more random draws (the first was not enough)
# space = c(rep("Train",17), rep("Val1", 1), rep("Val2", 1), rep("Test",1))
# set.seed(20)
# train_test_split1 = sample(space, 10^8, replace = TRUE)
# train_test_split = c(train_test_split, train_test_split1)
# save(train_test_split, file = "Data/Manual_data/Random_sequence_long.Rdata")
# load("Data/Manual_data/Random_sequence_long.Rdata")

# ==== Pipeline ====
pipeline = function(x, name, lang){
  # Check if name already exists
  test = any(grepl(name, list.files("Data/Training_data")))
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
    Keep_only_relevant()
  
  cat("\nSaving data")
  x %>% 
    Save_train_val_test(name, lang)
  cat("\nSaved:", name, lang)
  
  return(name)
}

# ==== Run pipelines DK ====
lang = "da"
x = pipeline(
  "Data/Tmp_data/Clean_DK_census.Rdata",
  "DK_census",
  lang
)

x = pipeline(
  "Data/Tmp_data/Clean_DK_cedar_translation.Rdata",
  "DK_cedar",
  lang
)

x = pipeline(
  "Data/Tmp_data/Clean_DK_orsted.Rdata",
  "DK_orsted",
  lang
)

# ==== Run pipelines EN ====
lang = "en"
x = pipeline(
  "Data/Tmp_data/Clean_EN_marr_cert.Rdata",
  "EN_marr_cert",
  lang
)

x = pipeline(
  "Data/Tmp_data/Clean_EN_parish_records.Rdata",
  "EN_parish",
  lang
)

x = pipeline(
  "Data/Tmp_data/Clean_LOC_EN.Rdata",
  "EN_loc",
  lang
)

x = pipeline(
  "Data/Tmp_data/Clean_O_CLACK.Rdata",
  "EN_oclack",
  lang
)

x = pipeline(
  "Data/Tmp_data/Clean_UK_IPUMS.Rdata",
  "EN_uk_ipums",
  lang
)

x = pipeline(
  "Data/Tmp_data/Clean_USA_IPUMS.Rdata",
  "EN_us_ipums",
  lang
)

# Canada
x = pipeline(
  "Data/Tmp_data/Clean_CA_IPUMS.Rdata",
  "EN_ca_ipums",
  "unk" # Unknown language (french and english mix)
)

# ==== Run pipelines NL ====
lang = "nl"
# Dutch data
x = pipeline(
  "Data/Tmp_data/Clean_HSN_database.Rdata",
  "HSN_database",
  lang
)

x = pipeline(
  "Data/Tmp_data/Clean_JIW.Rdata",
  "JIW_database",
  lang
)

# ==== Run pipelines SE ====
lang = "se"

x = pipeline(
  "Data/Tmp_data/Clean_SE_chalmers.Rdata",
  "SE_chalmers",
  lang
)

x = pipeline(
  "Data/Tmp_data/Clean_CEDAR_SE.Rdata",
  "SE_cedar",
  lang
)

x = pipeline(
  "Data/Tmp_data/Clean_SWEDPOP_SE.Rdata",
  "SE_swedpop",
  lang
)

# ==== Run pipelines NO ====
lang = "no"

x = pipeline(
  "Data/Tmp_data/Clean_NO_IPUMS.Rdata",
  "NO_ipums",
  lang
)

# ==== Run pipelines FR ====
lang = "fr"

x = pipeline(
  "Data/Tmp_data/Clean_French_desc.Rdata",
  "FR_desc",
  lang
)

# ==== Run pipelines Catalan ====
lang = "ca"

x = pipeline(
  "Data/Tmp_data/Clean_CA_BCN.Rdata",
  "CA_bcn",
  lang
)

# ==== Run pipelines DE ====
lang = "de"

x = pipeline(
  "Data/Tmp_data/Clean_DE_IPUMS.Rdata",
  "DE_ipums",
  lang
)

# ==== Run pipelines Iceland ====
lang = "is"

x = pipeline(
  "Data/Tmp_data/Clean_IS_IPUMS.Rdata",
  "IS_ipums",
  lang
)

# ==== Run pipelines IT ====
lang = "it"

x = pipeline(
  "Data/Tmp_data/Clean_IT_FM.Rdata",
  "IT_fm",
  lang
)

# ==== Training data stats ====
 
fs = list.files("Data/Training_data", pattern = ".csv", full.names = TRUE)
fs0 = list.files("Data/Training_data", pattern = ".csv")
x = foreach(f = fs, .combine = "bind_rows") %do% {
  x = read_csv(f) %>% NROW()
  data.frame(n = x)
} %>% mutate(f = fs0)


x = x %>% 
  arrange(n) %>% 
  mutate(
    all_n = round(n * 1/0.85)
  ) %>% 
  mutate(pct = round(100*n/sum(n), 3)) %>% 
  mutate(holdout = all_n - n)

bign = x$n %>% sum()
bigN = x$all_n %>% sum()
cat("\n", round(bign/1000000,3), "mil. training observations of", round(bigN/1000000,3), "mil. in total")
print(x)
