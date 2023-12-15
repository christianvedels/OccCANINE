# Data from HISCO website
# Created:    2023-12-15
# Auhtors:    Christian Vedel [christian-vs@sam.sdu.dk],
#
# Purpose:    This cleans training ship data from 
#             https://reshare.ukdataservice.ac.uk/853251/
#
# Output:     Clean tmp version of the data

# ==== Libraries =====
library(tidyverse)
source("Data_cleaning_scripts/000_Functions.R")
library(foreach)
library(haven)

# ==== Read data ====
data0 = read_dta("Data/Raw_data/Training_ship_data/indy_early.dta")
data1 = read_dta("Data/Raw_data/Training_ship_data/indy_late.dta")

data0 = data1 %>% bind_rows(data0)

# ==== Cleaning data0 ====
data0 = data0 %>% 
  select(f_occ, m_occ, f_occ_hisco, m_occ_hisco, ad_year)
  
data0_m = data0 %>% # Mother
  select(m_occ, m_occ_hisco, ad_year) %>% 
  filter(m_occ != "") %>% 
  rename(
    occ1 = m_occ,
    hisco_1 = m_occ_hisco
  ) %>% 
  mutate(
    gender = "female"
  )

data0_f = data0 %>% # Farther
  select(f_occ, f_occ_hisco, ad_year) %>% 
  filter(f_occ != "") %>% 
  rename(
    occ1 = f_occ,
    hisco_1 = f_occ_hisco
  ) %>% 
  mutate(
    gender = "male"
  )

data0 = data0_m %>% 
  bind_rows(data0_f) %>% 
  mutate(
    hisco_1 = ifelse(is.na(hisco_1), -1, hisco_1)
  )

data0 = data0 %>% 
  mutate( # Clean string:
    occ1 = str_replace_all(occ1, "[^[:alnum:] ]", "") %>% tolower()
  ) %>% 
  mutate(
    hisco_1 = as.numeric(hisco_1)
  ) %>% 
  mutate(
    hisco_1 = ifelse(is.na(hisco_1), -1, hisco_1)
  ) %>% 
  rename(
    Year = ad_year
  )

data0 = data0 %>%
  mutate(
    hisco_2 = " ",
    hisco_3 = " ",
    hisco_4 = " ",
    hisco_5 = " "
  )

combinations = Combinations(data0 %>% select(-gender), and = "and")

combinations = combinations %>%
  mutate_all(as.character)

data1 = data0 %>% 
  mutate_all(as.character) %>% 
  bind_rows(combinations)

# ==== Check against authoritative HISCO list ====
load("Data/Key.Rdata")

key = key %>% select(hisco, code)

# Remove data not in key (erronoeous data somehow)
data1 %>% 
  filter(!hisco_1 %in% key$hisco)

n1 = NROW(data1)
data1 = data1 %>% 
  filter(hisco_1 %in% key$hisco) %>% 
  filter(hisco_2 %in% key$hisco) %>% 
  filter(hisco_3 %in% key$hisco) %>% 
  filter(hisco_4 %in% key$hisco) %>% 
  filter(hisco_5 %in% key$hisco)

NROW(data1) - n1 # -38 observations

# Turn into character
data1 = data1 %>% 
  mutate_all(as.character)

# Add code
data1 = data1 %>% 
  left_join(
    key, by = c("hisco_1" = "hisco")
  ) %>% 
  rename(code1 = code) %>% 
  left_join(
    key, by = c("hisco_2" = "hisco")
  ) %>% 
  rename(code2 = code) %>% 
  left_join(
    key, by = c("hisco_3" = "hisco")
  ) %>% 
  rename(code3 = code) %>% 
  left_join(
    key, by = c("hisco_4" = "hisco")
  ) %>% 
  rename(code4 = code) %>% 
  left_join(
    key, by = c("hisco_5" = "hisco")
  ) %>% 
  rename(code5 = code)

# Add RowID 
data1 = data1 %>% 
  ungroup() %>% 
  mutate(RowID = 1:n())

# ==== Save ====
save(data1, file = "Data/Tmp_data/Clean_training_ship.Rdata")

