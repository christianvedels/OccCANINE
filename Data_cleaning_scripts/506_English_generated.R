# Generated English titles
# Updated:    2025-06-16
# Auhtors:    Christian Vedel [christian-vs@sam.sdu.dk],
#
# Purpose:    Cleans the data to get it ready to test performance

# ==== Libraries ====
library(tidyverse)
source("Data_cleaning_scripts/000_Functions.R")

# ==== Read data ====
data0 = read_csv("Data/Raw_data/2412_New_data/EN_titles_generated/historical_occupations.csv")

# ==== Cleaning data0 ====
data0 = data0 %>%
  rename(
    occ1 = Occupation
  ) %>%
  select(occ1)

data0 = data0 %>% 
  mutate( # Clean string:
    occ1 = str_replace_all(occ1, "[^[:alnum:] ]", "") %>% tolower()
  )

data1 = data0 %>%
  ungroup()

# Turn into character
data1 = data1 %>% 
  mutate_all(as.character)

# Add RowID 
data1 = data1 %>% 
  ungroup() %>% 
  mutate(
    lang = "en"
  )

# ==== Save ====
data1 %>% write_csv0("Data/OOD_data/EN_generated_titles.csv")
