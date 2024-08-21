# PSTI_v3 
# Updated:    2024-08-15
# Auhtors:    Christian Vedel [christian-vs@sam.sdu.dk],
#
# Purpose:    Processes the PSTI_v3 data
#
# Output:     Clean version

# ==== Libraries ====
library(tidyverse)
source("Data_cleaning_scripts/000_Functions.R")

# ==== Load data ====
data0 = read_csv("Data/Raw_data/2404_New_data/PSTI_v3/ENG_PST_STRINGS_coded01022024.csv")

# ==== Data cleaning ====
# Extract both raw and standardized strings
data1 = data0 %>% 
  select(`Original occupation`, PSTI_V3) %>% 
  rename(occ1 = `Original occupation`)

data2 = data0 %>% 
  select(`Standardised occupation`, PSTI_V3) %>% 
  rename(occ1 = `Standardised occupation`) %>% 
  distinct()

data1 = bind_rows(data1, data2)

# Split multiple occupations
data1 = data1 %>% 
  mutate(nchar0 = nchar(PSTI_V3)) %>% 
  mutate(
    multiple = grepl(";", PSTI_V3)
  ) %>%
  # filter(
  #   multiple
  # ) %>% 
  mutate(
    PSTI_1 = PSTI_V3
  ) %>% 
  rowwise() %>% 
  mutate(
    tmp = strsplit(PSTI_1, ";")
  ) %>% 
  mutate(
    PSTI_1 = tmp[1],
    PSTI_2 = tmp[2],
    PSTI_3 = tmp[3],
    PSTI_4 = tmp[4],
    PSTI_5 = tmp[5],
  ) %>% 
  select(-tmp) %>% 
  ungroup() %>% 
  drop_na(occ1)

# ==== Synthetic combinations ====
# Renaming temporarily to use 'Combinations' function
data_single_occ = data1 %>%
  filter(!multiple) %>% 
  rename(
    hisco_1 = PSTI_1,
    hisco_2 = PSTI_2,
    hisco_3 = PSTI_3,
    hisco_4 = PSTI_4,
    hisco_5 = PSTI_5
  )

set.seed(20)
data2 = data_single_occ %>% 
  Combinations() %>% 
  rename(
    PSTI_1 = hisco_1,
    PSTI_2 = hisco_2,
    PSTI_3 = hisco_3,
    PSTI_4 = hisco_4,
    PSTI_5 = hisco_5
  ) %>% 
  mutate(
    synthetic_combination = 1
  )

data1 = data1 %>% 
  bind_rows(data2) %>% 
  mutate(
    synthetic_combination = ifelse(is.na(synthetic_combination), 0, synthetic_combination)
  ) %>% 
  mutate_all( # Make NA " "
    function(x){
      ifelse(is.na(x), " ", x)
    }
  ) %>% 
  ungroup() %>% 
  mutate(RowID = 1:n()) %>% 
  mutate(
    occ1 = tolower(occ1)
  )

# ==== Save ====
save(data1, file = "Data/Tmp_data/EN_PSTI.Rdata")
  
