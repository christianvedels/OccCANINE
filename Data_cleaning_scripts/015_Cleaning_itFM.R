# Cleaning Italian Formasin Marzona data (FM data)
# Created:    2023-11-08
# Auhtors:    Christian Vedel [christian-vs@sam.sdu.dk],
#
# Purpose:    This script cleans occupations from the It_FM data 
#             https://hdl.handle.net/10622/SRVW6S
# Output:     Clean tmp version of the inputted data

# ==== Libraries =====
library(tidyverse)
source("Data_cleaning_scripts/000_Functions.R")
library(foreach)

# ==== Read data ====
data0 = read_csv("Data/Raw_data/Italian Formasin Marzona/HISCO_Italian_Formasin_Marzona_2006.csv")

# ==== Cleaning data0 ====
data0 = data0 %>% 
  rename(
    occ1 = occupation,
    hisco_1 = hisco,
  ) %>% 
  mutate( # Clean string:
    occ1 = str_replace_all(occ1, "[^[:alnum:] ]", "") %>% tolower()
  ) %>% 
  mutate(
    hisco_1 = as.numeric(hisco_1)
  ) %>% 
  mutate(
    hisco_1 = ifelse(is.na(hisco_1), -1, hisco_1)
  ) %>% 
  select(occ1, hisco_1)

data0 = data0 %>%
  mutate(
    hisco_2 = " ",
    hisco_3 = " ",
    hisco_4 = " ",
    hisco_5 = " "
  )

combinations = Combinations(data0, and = "e")

combinations = combinations %>%
  mutate(
    hisco_3 = " ",
    hisco_4 = " ",
    hisco_5 = " "
  ) %>% 
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

NROW(data1) - n1 # -0 observations

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
save(data1, file = "Data/Tmp_data/Clean_IT_FM.Rdata")
