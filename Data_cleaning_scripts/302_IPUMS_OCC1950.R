# OCC1950 IPUMS 
# Updated:    2024-08-15
# Auhtors:    Christian Vedel [christian-vs@sam.sdu.dk],
#
# Purpose:    Processes the US IPUMS samples
#
# Output:     Clean version

# ==== Libraries ====
library(tidyverse)
source("Data_cleaning_scripts/000_Functions.R")

# ==== Load data ====
data0 = read_csv("Data/Raw_data/2404_New_data/IPUMS_USA_samples/usa_00009.csv")

# ==== Data cleaning ====
# Print view
data0 %>% 
  select(starts_with("occ"))

# Fix string
data0 = data0 %>% 
  drop_na(OCCSTR) %>% 
  rename(
    occ1 = OCCSTR
  ) %>% 
  mutate(
    occ1 = iconv(occ1, from = "UTF-8", to = "UTF-8", sub = "")
  ) %>% 
  mutate(
    occ1 = tolower(occ1)
  ) %>% 
  rename(
    Year = YEAR
  ) %>% 
  select(-OCC) %>%
  rename(
    OCC1950_1 = OCC1950
  )

# ==== Synthetic combinations ====
# Renaming temporarily to use 'Combinations' function
data1 = data0 %>% 
  rename(
    hisco_1 = OCC1950_1
  ) %>% 
  mutate(
    hisco_2 = " "
  )

set.seed(20)
data2 = data1 %>% 
  sample_n(200000) %>% 
  Combinations() %>% 
  rename(
    OCC1950_1 = hisco_1,
    OCC1950_2 = hisco_2
  ) %>% 
  mutate(
    synthetic_combination = 1
  )

data1 = data0 %>% 
  bind_rows(data2) %>% 
  mutate(
    synthetic_combination = ifelse(is.na(synthetic_combination), 0, synthetic_combination)
  ) %>% 
  ungroup() %>% 
  mutate(RowID = 1:n()) %>% 
  select(
    occ1,
    OCC1950_1,
    OCC1950_2,
    synthetic_combination
  ) %>% 
  mutate_all( # Make NA " "
    function(x){
      ifelse(is.na(x), " ", x)
    }
  ) %>% 
  mutate(RowID = 1:n())

# ==== Save ====
save(data1, file = "Data/Tmp_data/EN_IPUMS_OCC1950.Rdata")

