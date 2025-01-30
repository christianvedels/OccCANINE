# Cleans the bankruptcy data from Korn and Lacroix
# Updated:    2025-01-30
# Auhtors:    Christian Vedel [christian-vs@sam.sdu.dk],
#
# Purpose:    Cleans the data to get it ready to test performance

# ==== Libraries ====
library(tidyverse)
source("Data_cleaning_scripts/000_Functions.R")

# ==== Read data ====
data0 = read_csv("Data/Raw_data/2412_New_data/Bankruptcy_KornLacroix/Bankruptcy_KornLacroix.csv")

# ==== Cleaning data0 ====
# Pivot longer
data0 = data0 %>%
  mutate_all(as.character) %>%
  pivot_longer(
    cols = starts_with("occupation"),
    names_to = "occupation_no",
    values_to = "occupation"
  ) %>%
  pivot_longer(
    cols = starts_with("HISCO"),
    names_to = "HISCO_no",
    values_to = "HISCO"
  ) %>%
  filter(
    str_extract(occupation_no, "\\d+") == str_extract(HISCO_no, "\\d+"),
    !is.na(occupation)
  ) %>%
  arrange(as.numeric(unique_id))

data0 = data0 %>%
  rename(
    occ1 = occupation,
    hisco_1 = HISCO
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
  mutate(not_in_key = hisco_1 %in% key$hisco) %>% 
  mutate(not_in_key = not_in_key + hisco_2 %in% key$hisco) %>%
  mutate(not_in_key = not_in_key + hisco_3 %in% key$hisco) %>%
  mutate(not_in_key = not_in_key + hisco_4 %in% key$hisco) %>%
  mutate(not_in_key = not_in_key + hisco_5 %in% key$hisco)

# Turn into character
data1 = data1 %>% 
  mutate_all(as.character)

# Add RowID 
data1 = data1 %>% 
  ungroup() %>% 
  mutate(RowID = 1:n())

# ==== Save ====
data1 %>% write_csv0("Data/OOD_data/EN_Bankruptcy_KornLacroix.csv")
