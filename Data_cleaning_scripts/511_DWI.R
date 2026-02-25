# Cleaning Danish West Indies data
# Created:    2024-08-27
# Authors:    Christian Vedel [christian-vs@sam.sdu.dk],
#
# Purpose:    Cleans the data to get it ready to test performance

# ==== Libraries ====
library(tidyverse)
library(haven)
source("Data_cleaning_scripts/000_Functions.R")

# ==== Read data ====
data0 = read_dta("Data/Raw_data/2412_New_data/Danish_west_indies/Censuspanel.dta/Censuspanel.dta")

# ==== Select relevant variables ====
data0 = data0 %>%
    select(Occupation, Occupation_spelling_stand, st_occupation, HISCO)

# ==== Cleaning data0 ====
data1 = data0 %>%
  select(Occupation, HISCO) %>%
  rename(
    occ1 = Occupation,
    hisco_1 = HISCO
  ) %>%
  mutate(
    hisco_2 = " ",
    hisco_3 = " ",  
    hisco_4 = " ",
    hisco_5 = " "
  ) %>%
  mutate(
    hisco_1 = ifelse(is.na(hisco_1), "-1", hisco_1)
  ) %>%
  mutate(hisco_1 = ifelse(hisco_1 %in% c("0", "2", "3", "4", "5", "9"), "-1", hisco_1))

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
  mutate(not_in_key = not_in_key + !(hisco_5 %in% key$hisco)) %>%
  filter(not_in_key == 0) %>%
  select(-not_in_key) %>%
  ungroup()

n2 = NROW(data1)
print(paste0("Removed ", n1 - n2, " rows not in key"))

# Turn into character
data1 = data1 %>% 
  mutate_all(as.character)

# Add RowID 
data1 = data1 %>% 
  ungroup() %>% 
  mutate(RowID = 1:n()) %>%
  mutate(
    lang = "en"
  )

# Length of HISCO codes (leading zeros)
data1 = data1 %>%
    mutate(
        hisco_1 = ifelse(is.na(hisco_1) | hisco_1 == "-1" | hisco_1 == " ", hisco_1, str_pad(hisco_1, 5, pad = "0")),
        hisco_2 = ifelse(is.na(hisco_2) | hisco_2 == "-1" | hisco_2 == " ", hisco_2, str_pad(hisco_2, 5, pad = "0")),
        hisco_3 = ifelse(is.na(hisco_3) | hisco_3 == "-1" | hisco_3 == " ", hisco_3, str_pad(hisco_3, 5, pad = "0")),
        hisco_4 = ifelse(is.na(hisco_4) | hisco_4 == "-1" | hisco_4 == " ", hisco_4, str_pad(hisco_4, 5, pad = "0")),
        hisco_5 = ifelse(is.na(hisco_5) | hisco_5 == "-1" | hisco_5 == " ", hisco_5, str_pad(hisco_5, 5, pad = "0"))
    )

# ==== Save ====
data1 %>% write_csv0("Data/OOD_data/EN_dwi.csv")
