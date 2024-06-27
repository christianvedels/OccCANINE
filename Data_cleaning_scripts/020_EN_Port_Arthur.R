# PortArthur occupations
# Created:    2024-06-18
# Auhtors:    Christian Vedel [christian-vs@sam.sdu.dk],
#
# Purpose:    Clean Port Arthur penal station HISCO codes
#
# Output:     Clean tmp version of the data

# ==== Libraries =====
library(tidyverse)
source("Data_cleaning_scripts/000_Functions.R")
library(foreach)

# ==== Read data ====
data0 = read_csv2("Data/Raw_data/2404_New_data/Port_Arthur_penal_station/PortArthur.csv")

# ==== Cleaning data0 ====
data0 = data0 %>% 
  rename(
    occ1 = `Port Arthur occupation`,
    hisco_1 = `HISCO Code`
  ) %>% 
  select(occ1, hisco_1) %>% 
  drop_na()

# ==== More cleaning ====
data0 = data0 %>% 
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

set.seed(20)
combinations = Combinations(data0, and = "and")

combinations = combinations %>%
  mutate_all(as.character)

data1 = data0 %>% 
  mutate_all(as.character) %>% 
  bind_rows(combinations)

# Remove cases with 2 of the same occupation
data1 = data1 %>% 
  filter(hisco_1 != hisco_2)

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

NROW(data1) - n1 # 0 observations

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
save(data1, file = "Data/Tmp_data/PortArthur.Rdata")

