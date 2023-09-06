# Cleaning Chalmers Orster data
# Created:    2023-09-06
# Auhtors:    Christian Vedel [christian-vs@sam.sdu.dk],
#
# Purpose:    This script cleans data from Nick Fords Chalmers/Ørsted paper
#
# Output:     Clean tmp version of the inputted data

# ==== Libraries =====
library(tidyverse)
source("Data_cleaning_scripts/000_Functions.R")

# ==== Read data ====
DK_cedar = read_csv2("Data/Raw_data/Chalmers_Orsted/DK - se_CEDAR_HISCO.csv")
SEDK_data = read_csv("Data/Raw_data/Chalmers_Orsted/Vedel - HISCO summary.csv")

# ==== Cleaning DK cedar translation ====
# Standardizing strings
DK_cedar = DK_cedar %>%
  mutate(
    HISCO = as.character(HISCO)
  ) %>% 
  mutate( # Clean string:
    occ1 = str_replace_all(occupation, "[^[:alnum:] ]", "") %>% tolower()
  ) %>% 
  rename(
    hisco_1 = HISCO
  ) %>% 
  select(occ1, hisco_1) %>% 
  mutate(
    hisco_1 = as.numeric(hisco_1)
  ) %>%
  mutate( # Remove scandi letters
    occ1 = occ1 %>% sub_scandi()
  )

DK_cedar = DK_cedar %>% as.data.frame()

# NA padding
DK_cedar = DK_cedar %>%
  mutate(
    hisco_2 = " ",
    hisco_3 = " ",
    hisco_4 = " ",
    hisco_5 = " "
  )

# ==== Check DK cedar against authorative HISCO list ====
load("Data/Key.Rdata")

key = key %>% select(hisco, code)

# Remove data not in key (erronoeous data somehow)
DK_cedar %>% 
  filter(!hisco_1 %in% key$hisco)

n1 = NROW(DK_cedar)
DK_cedar = DK_cedar %>% 
  filter(hisco_1 %in% key$hisco) %>% 
  filter(hisco_2 %in% key$hisco) %>% 
  filter(hisco_3 %in% key$hisco) %>% 
  filter(hisco_4 %in% key$hisco) %>% 
  filter(hisco_5 %in% key$hisco)

NROW(DK_cedar) - n1 # 85 observations

# Upsampling
set.seed(20)
DK_cedar = upsample(DK_cedar, 10)

# Add RowID 
DK_cedar = DK_cedar %>% 
  ungroup() %>% 
  mutate(RowID = 1:n())

# ==== Cleaning SE chalmers data ====
# Standardizing strings
SE_chalmers = SEDK_data %>% 
  filter(all_land == "SE")

SE_chalmers = SE_chalmers %>%
  pivot_longer(
    f_occ:hisco_occ
  ) %>% 
  mutate(
    hisco = as.character(hisco)
  ) %>% 
  mutate( # Clean string:
    occ1 = str_replace_all(value, "[^[:alnum:] ]", "") %>% tolower()
  ) %>% 
  rename(
    hisco_1 = hisco
  ) %>% 
  select(occ1, hisco_1) %>% 
  mutate(
    hisco_1 = as.numeric(hisco_1)
  ) %>%
  mutate( # Remove scandi letters
    occ1 = occ1 %>% sub_scandi()
  )

# NA padding
SE_chalmers = SE_chalmers %>%
  mutate(
    hisco_2 = " ",
    hisco_3 = " ",
    hisco_4 = " ",
    hisco_5 = " "
  )

# Remove data not in key (erronoeous data somehow)
SE_chalmers %>% 
  filter(!hisco_1 %in% key$hisco)

n1 = NROW(SE_chalmers)
SE_chalmers = SE_chalmers %>% 
  filter(hisco_1 %in% key$hisco) %>% 
  filter(hisco_2 %in% key$hisco) %>% 
  filter(hisco_3 %in% key$hisco) %>% 
  filter(hisco_4 %in% key$hisco) %>% 
  filter(hisco_5 %in% key$hisco)

NROW(SE_chalmers) - n1 # 0 observations
# Upsampling
set.seed(20)
SE_chalmers = upsample(SE_chalmers, 10)

# Add rowID
SE_chalmers = SE_chalmers %>% 
  mutate(RowID = 1:n())

# ==== Cleaning DK orsted data ====
# Standardizing strings
DK_orsted = SEDK_data %>% 
  filter(all_land == "DK")

DK_orsted = DK_orsted %>%
  pivot_longer(
    f_occ:hisco_occ
  ) %>% 
  mutate(
    hisco = as.character(hisco)
  ) %>% 
  mutate( # Clean string:
    occ1 = str_replace_all(value, "[^[:alnum:] ]", "") %>% tolower()
  ) %>% 
  rename(
    hisco_1 = hisco
  ) %>% 
  select(occ1, hisco_1) %>% 
  mutate(
    hisco_1 = as.numeric(hisco_1)
  ) %>%
  mutate( # Remove scandi letters
    occ1 = occ1 %>% sub_scandi()
  )

# NA padding
DK_orsted = DK_orsted %>%
  mutate(
    hisco_2 = " ",
    hisco_3 = " ",
    hisco_4 = " ",
    hisco_5 = " "
  )

# Remove data not in key (erronoeous data somehow)
DK_orsted %>% 
  filter(!hisco_1 %in% key$hisco)

n1 = NROW(DK_orsted)
DK_orsted = DK_orsted %>% 
  filter(hisco_1 %in% key$hisco) %>% 
  filter(hisco_2 %in% key$hisco) %>% 
  filter(hisco_3 %in% key$hisco) %>% 
  filter(hisco_4 %in% key$hisco) %>% 
  filter(hisco_5 %in% key$hisco)

NROW(DK_orsted) - n1 # 0 observations

# Upsampling
set.seed(20)
DK_orsted = upsample(DK_orsted, 10)

# Add rowID
DK_orsted = DK_orsted %>% 
  mutate(RowID = 1:n())

# ==== Save ====
save(DK_cedar, file = "Data/Tmp_data/Clean_DK_cedar_translation.Rdata")
save(DK_orsted, file = "Data/Tmp_data/Clean_DK_orsted.Rdata")
save(SE_chalmers, file = "Data/Tmp_data/Clean_SE_chalmers.Rdata")
