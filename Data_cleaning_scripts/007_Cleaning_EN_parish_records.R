# Cleaning English Parish Records
# Created:    2023-09-06
# Auhtors:    Christian Vedel [christian-vs@sam.sdu.dk],
#
# Purpose:    This script cleans data from English Parish Records
#
# Output:     Clean tmp version of the inputted data

# ==== Libraries =====
library(tidyverse)
source("Data_cleaning_scripts/000_Functions.R")

# ==== Read data ====
data0 = haven::read_dta("Data/Raw_data/English Parish Records/Occs_Hisco.dta")

# ==== Removing customs codes in this data ====
data0 = data0 %>% 
  filter(hcode != 12345)

# ==== Upsamples based on frequency ====
warning("Upsampling based on data frequencies would make sense")

# ==== Cleaning data0 ====

data0 = data0 %>%
  pivot_longer(
    occ:std_occ
  ) %>% 
  rename(
    hisco_1 = hcode,
    occ1 = value
  ) %>% 
  mutate( # Clean string:
    occ1 = str_replace_all(occ1, "[^[:alnum:] ]", "") %>% tolower()
  ) %>% 
  select(occ1, hisco_1) %>% 
  mutate(
    hisco_1 = as.numeric(hisco_1)
  )

data0 = data0 %>%
  mutate(
    hisco_2 = " ",
    hisco_3 = " ",
    hisco_4 = " ",
    hisco_5 = " "
  )

# ==== Check against authorative HISCO list ====
load("Data/Key.Rdata")

key = key %>% select(hisco, code)

# Remove data not in key (erronoeous data somehow)
data0 %>% 
  filter(!hisco_1 %in% key$hisco)

n1 = NROW(data0)
data0 = data0 %>% 
  filter(hisco_1 %in% key$hisco) %>% 
  filter(hisco_2 %in% key$hisco) %>% 
  filter(hisco_3 %in% key$hisco) %>% 
  filter(hisco_4 %in% key$hisco) %>% 
  filter(hisco_5 %in% key$hisco)

NROW(data0) - n1 # 34 observations

# Add RowID 
data0 = data0 %>% 
  ungroup() %>% 
  mutate(RowID = 1:n())

# ==== Save ====
save(data0, file = "Data/Tmp_data/Clean_EN_parish_records.Rdata")
