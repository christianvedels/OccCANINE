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
data0 = read_csv("Data/Raw_data/2404_New_data/IPUMS_UK/ipumsi_00005.csv")

# ==== Data cleaning ====
# Fix string
data0 = data0 %>% 
  drop_na(OCCSTRNG) %>% 
  rename(
    occ1 = OCCSTRNG
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
  select(
    Year, occ1, ISCO68A, OCCICEM, 
  ) %>% 
  rename(
    ISCO68A_1 = ISCO68A, 
    OCCICEM_1 = OCCICEM
  )
 
# ==== Synthetic combinations OCCICEM ====
# Renaming temporarily to use 'Combinations' function
data1 = data0 %>% 
  rename(
    hisco_1 = OCCICEM_1
  ) %>% 
  mutate(
    hisco_2 = " "
  )

set.seed(20)
data2 = data1 %>% 
  sample_n(200000) %>% 
  Combinations() %>% 
  rename(
    OCCICEM_1 = hisco_1,
    OCCICEM_2 = hisco_2
  ) %>% 
  mutate(
    synthetic_combination = 1
  ) %>% 
  filter(
    OCCICEM_1 != OCCICEM_2
  )

data1_occicem = data0 %>% 
  bind_rows(data2) %>% 
  mutate(
    synthetic_combination = ifelse(is.na(synthetic_combination), 0, synthetic_combination)
  ) %>% 
  ungroup() %>% 
  mutate(RowID = 1:n()) %>% 
  select(
    occ1,
    OCCICEM_1,
    OCCICEM_2,
    synthetic_combination
  ) %>% 
  mutate_all( # Make NA " "
    function(x){
      ifelse(is.na(x), " ", x)
    }
  ) %>% 
  mutate(RowID = 1:n())

# ==== Synthetic combinations ISCO68 ====
# Renaming temporarily to use 'Combinations' function
data1 = data0 %>% 
  rename(
    hisco_1 = ISCO68A_1
  ) %>% 
  mutate(
    hisco_2 = " "
  )

set.seed(20)
data2 = data1 %>% 
  sample_n(2000000) %>% 
  Combinations() %>% 
  rename(
    ISCO68A_1 = hisco_1,
    ISCO68A_2 = hisco_2
  ) %>% 
  mutate(
    synthetic_combination = 1
  ) %>% 
  filter(
    ISCO68A_1 != ISCO68A_2
  )

data1_isco68 = data0 %>% 
  bind_rows(data2) %>% 
  mutate(
    synthetic_combination = ifelse(is.na(synthetic_combination), 0, synthetic_combination)
  ) %>% 
  ungroup() %>% 
  mutate(RowID = 1:n()) %>% 
  select(
    occ1,
    ISCO68A_1,
    ISCO68A_2,
    synthetic_combination
  ) %>% 
  mutate_all( # Make NA " "
    function(x){
      ifelse(is.na(x), " ", x)
    }
  ) %>% 
  mutate(RowID = 1:n())

# ==== Save ====
save(data1_occicem, file = "Data/Tmp_data/EN_IPUMS_UK_OCCICEM.Rdata")
save(data1_isco68, file = "Data/Tmp_data/EN_IPUMS_UK_OCCICEM.Rdata")
