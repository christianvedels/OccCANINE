# Dutch famiegeld data
# Updated:    2025-01-30
# Auhtors:    Christian Vedel [christian-vs@sam.sdu.dk],
#
# Purpose:    Cleans the data to get it ready to test performance

# ==== Libraries ====
library(tidyverse)
library(readxl)
source("Data_cleaning_scripts/000_Functions.R")

# ==== Read data ====
data0 = read_excel("Data/Raw_data/2412_New_data/Familie_Geld_Brams_corrections/Familiegeld_sample_Bram.xls")

# ==== Cleaning data0 ====
data0 = data0 %>%
  select(-occ1) %>%
  rename(
    occ1 = occupation,
    hisco_1 = hisco_1
  ) %>%
  mutate(
    hisco_1 = ifelse(!is.na(bram_hisco), bram_hisco, hisco_1)
  ) %>%
  select(occ1, hisco_1)

data0 = data0 %>% 
  mutate( # Clean string:
    occ1 = str_replace_all(occ1, "[^[:alnum:] ]", "") %>% tolower()
  ) %>% 
  mutate(
    hisco_1 = as.numeric(hisco_1)
  ) %>% 
  mutate(
    was_NA = as.numeric(is.na(hisco_1))
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

data1 = data0 %>%
  ungroup()

# ==== Check against authoritative HISCO list ====
load("Data/Key.Rdata")

key = key %>% select(hisco, code)

# Remove data not in key (erronoeous data somehow)
data1 %>% 
  filter(!hisco_1 %in% key$hisco)

n1 = NROW(data1)
data1 = data1 %>% 
  mutate(not_in_key = !(hisco_1 %in% key$hisco)) %>% 
  mutate(not_in_key = not_in_key + !(hisco_2 %in% key$hisco)) %>%
  mutate(not_in_key = not_in_key + !(hisco_3 %in% key$hisco)) %>%
  mutate(not_in_key = not_in_key + !(hisco_4 %in% key$hisco)) %>%
  mutate(not_in_key = not_in_key + !(hisco_5 %in% key$hisco))

# Turn into character
data1 = data1 %>% 
  mutate_all(as.character)

# Add RowID 
data1 = data1 %>% 
  ungroup() %>% 
  mutate(RowID = 1:n()) %>%
  mutate(
    lang = "SE"
  )

# ==== Save ====
data1 %>% write_csv0("Data/OOD_data/NL_familiegeld.csv")
