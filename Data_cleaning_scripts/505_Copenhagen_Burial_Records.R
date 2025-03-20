# Copen
# Updated:    2025-01-30
# Auhtors:    Christian Vedel [christian-vs@sam.sdu.dk],
#
# Purpose:    Cleans the data to get it ready to test performance

# ==== Libraries ====
library(tidyverse)
source("Data_cleaning_scripts/000_Functions.R")

# ==== Read data ====
data0 = read_csv("Data/Application_data/Copenhagen Burial Records/transcribed_sources/CBP/CBP_20210309.csv")

# ==== Cleaning data0 ====
data0 = data0 %>%
  rename(
    occ1 = positions
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
    lang = "DA"
  )

# Sample
set.seed(20)
data1 = data1 %>%
    drop_na(occ1) %>%
    sample_n(1000) %>%
    count(occ1) %>% 
    mutate(RowID = 1:n())


# ==== Save ====
data1 %>% write_csv0("Data/OOD_data/DA_Copenhagen_Burial_Records.csv")
