# Leeuwen data cleaning
# Created:    2024-06-18
# Updated:    2026-10-17
# Auhtors:    Christian Vedel [christian-vs@sam.sdu.dk],
#
# Purpose:    This script cleans HISCO coded titles from various sources
#             shared by Marco van Leeuwen
#
# Output:     Clean tmp version of the data

# ==== Libraries =====
library(tidyverse)
library(readxl)
library(foreach)
source("Data_cleaning_scripts/000_Functions.R")

# ==== Read data ====
# Belgium Dutch (Flemish, by us and Koen Matthijs and his team)
HISCOdata_correct_withstatus = read_excel("Data/Raw_data/2509_New_data/Leeuwen_various_sources/HISCOdata_correct_withstatus.xls")
HISCOdata_correct_withstatus$lang = "unk" # Mix of Flemish and French

# Belgian French (Walloon, for TOS)
BEL_French = read_excel("Data/Raw_data/2509_New_data/Leeuwen_various_sources/BEL_French_fromUKfile20012012.xls")
BEL_French$lang = "fr"

# Canada HIsco_Balsac (for the HISCO book by Michèle de Seve)
HISCO_BALSAC = read_excel("Data/Raw_data/2509_New_data/Leeuwen_various_sources/HISCO_BALSAC_nov2015.xlsx")
HISCO_BALSAC$lang = "unk" # Mix of French and English

# France (2012; mostly by Jean-Pierre Pélissier and Danièle Rébaudo)
france = read_excel("Data/Raw_data/2509_New_data/Leeuwen_various_sources/france.xls", sheet = "france")
france$lang = "fr"

# German (Germany TOS codes by us,  and Switzerland by Simon Seiler)
hisco = read_excel("Data/Raw_data/2509_New_data/Leeuwen_various_sources/Hisco.xlsx")
hisco$lang = "ge"

# Italy (by Carlo Corsini)
Professioni = read_excel("Data/Raw_data/2509_New_data/Leeuwen_various_sources/Professioni 1841, Hisco TUTTIX.xls")
Professioni$lang = "it"

# ==== Clean data ====
data1 = HISCOdata_correct_withstatus %>%
    rename(
        occ1 = profession1_groom,
        hisco_1 = HISCO_groom
    ) %>%
    select(occ1, hisco_1, lang) %>%
    mutate(source = "Leeuwen/HISCOdata_correct_withstatus") %>%
    mutate_all(as.character)

data2 = BEL_French %>%
    rename(
        occ1 = TITLE,
        occ2 = `CORRECTED TITLE`,
        hisco_1 = HISCOTXT3
    ) %>%
    pivot_longer(cols = c("occ1", "occ2"), names_to = "var", values_to = "occ1") %>%
    drop_na(occ1) %>%
    select(-var) %>%
    select(occ1, hisco_1, lang) %>%
    mutate(source = "Leeuwen/BEL_French") %>%
    mutate_all(as.character)

data3 = HISCO_BALSAC %>%
    rename(
        occ1 = descr,
        hisco_1 = hisco
    ) %>%
    select(occ1, hisco_1, lang) %>%
    mutate(source = "Leeuwen/HISCO_BALSAC") %>%
    mutate_all(as.character)

data4 = france %>%
    rename(
        occ1 = Title,
        hisco_1 = HISCO
    ) %>%
    select(occ1, hisco_1, lang) %>%
    mutate(source = "Leeuwen/france") %>%
    mutate_all(as.character)

data5 = hisco %>%
    rename(
        occ1 = title,
        occ2 = `standardized title`,
        hisco_1 = HISCO
    ) %>%
    pivot_longer(cols = c("occ1", "occ2"), names_to = "var", values_to = "occ1") %>%
    drop_na(occ1) %>%
    select(occ1, hisco_1, lang) %>%
    mutate(source = "Leeuwen/hisco") %>%
    mutate_all(as.character)

data6 = Professioni %>%
    rename(
        occ1 = Professione,
        hisco_1 = HiscoId
    ) %>%
    select(occ1, hisco_1, lang) %>%
    # Convert from 6-31.20 to 63120
    mutate(
        hisco_new = gsub("-", "", hisco_1),
        hisco_new = gsub("\\.", "", hisco_new),
        hisco_new = as.numeric(hisco_new)
    ) %>%
    mutate(
        hisco_1 = ifelse(hisco_1 == "-1", -1, hisco_new)
    ) %>%
    select(-hisco_new) %>%
    mutate(source = "Leeuwen/Professioni") %>%
    mutate_all(as.character)

data0 = bind_rows(data1, data2, data3, data4, data5, data6)

# ==== More cleaning ====
data0 = data0 %>% 
    mutate( # Clean string:
        occ1 = occ1 %>% tolower()
    ) %>% 
    mutate(
        hisco_1 = as.numeric(hisco_1)
    ) %>% 
    mutate(
        hisco_1 = ifelse(is.na(hisco_1), -1, hisco_1)
    ) %>%
    mutate(
        hisco_2 = " ",
        hisco_3 = " ",
        hisco_4 = " ",
        hisco_5 = " "
    )

# ==== Get combinations ====
set.seed(20)
sources = unique(data0$source)
all_data = foreach(s = sources, .combine = "bind_rows") %do% {
  cat("\nGetting combinations for", s)
  data_s = data0 %>% filter(source == s) %>% mutate_all(as.character)

  # No combinations for sources mixing languages (no single word for 'and')
  if(any(data_s$lang == "unk")){
    cat(" - skipped (mixed languages)")
    return(data_s)
  }

  # At most as many combinations as original rows (max 50k).
  # Combinations() returns 10 combinations per input row.
  n_s = NROW(data_s)
  n_comb = ceiling(min(50000, n_s) / 10)
  combinations = data_s %>%
    mutate_all(as.character) %>%
    filter(hisco_2 == " ", hisco_1 != "-1") %>%
    sample_n(n_comb, replace = TRUE) %>%
    Combinations("")

  data_s = data_s %>% bind_rows(combinations) 
  return(data_s)
}

# Remove cases with 2 of the same occupation
all_data = all_data %>% 
  filter(hisco_1 != hisco_2)

data0 = all_data

# ==== Check against authoritative HISCO list ====
load("Data/Key.Rdata")

key = key %>% select(hisco, code)

# Remove data not in key (erronoeous data somehow)
data0 %>% 
  filter(!hisco_1 %in% key$hisco)

data0 %>% 
  filter(!hisco_2 %in% key$hisco)

n1 = NROW(data0)
data0 = data0 %>% 
  filter(hisco_1 %in% key$hisco) %>% 
  filter(hisco_2 %in% key$hisco) %>% 
  filter(hisco_3 %in% key$hisco) %>% 
  filter(hisco_4 %in% key$hisco) %>% 
  filter(hisco_5 %in% key$hisco)

NROW(data0) - n1 # 0 observations

# Turn into character
data0 = data0 %>% 
  mutate_all(as.character)

# Add code
data0 = data0 %>% 
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
data0 = data0 %>% 
  ungroup() %>% 
  mutate(RowID = 1:n())

# ==== Save ====
save(data0, file = "Data/Tmp_data/Leeuwen_data.Rdata")

