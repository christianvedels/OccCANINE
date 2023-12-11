# Cleaning JIW https://datarepository.eur.nl/articles/dataset/Dataset_Ja_ik_wil_-_Amsterdam_marriage_banns_registers_1580-1810/14049842
# Created:    2023-11-08
# Auhtors:    Christian Vedel [christian-vs@sam.sdu.dk],
#
# Purpose:    This script cleans occupations from the 'Ja, ik will' database
#
# Output:     Clean tmp version of the inputted data

# ==== Libraries =====
library(tidyverse)
source("Data_cleaning_scripts/000_Functions.R")
library(foreach)
library(readxl)

# ==== Read data ====
data0 = read_excel("Data/Raw_data/JIW data/JIW data_version-20210217.xlsx")

# ==== Cleaning data0 ====
data0 = data0 %>%
  select(
    groom_occupation_current,
    bruidegom_beroep,
    `Groom_occupation-HIScode`,
    reg_year
  ) %>% 
  pivot_longer(
    cols = groom_occupation_current:bruidegom_beroep,
    names_to = "type"
  ) %>% 
  mutate(
    reg_year = ifelse(type == "groom_occupation_current", NA, reg_year)
  ) %>% 
  rename(
    occ1 = value,
    hisco_1 = `Groom_occupation-HIScode`,
    Year = "reg_year"
  ) %>% 
  mutate( # Clean string:
    occ1 = str_replace_all(occ1, "[^[:alnum:] ]", "") %>% tolower()
  ) %>% 
  mutate(
    hisco_1 = as.numeric(hisco_1)
  ) %>% 
  mutate(
    hisco_1 = ifelse(is.na(hisco_1), -1, hisco_1)
  )

data0 = data0 %>%
  mutate(
    hisco_2 = " ",
    hisco_3 = " ",
    hisco_4 = " ",
    hisco_5 = " "
  )

combinations = Combinations(data0, and = "en")

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

NROW(data1) - n1 # -9801 observations

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
save(data1, file = "Data/Tmp_data/Clean_JIW.Rdata")
