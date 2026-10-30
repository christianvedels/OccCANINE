# Mosaic data cleaning script
# Created:    2024-09-24
# Updated:    2026-10-17
# Auhtors:    Christian Vedel [christian-vs@sam.sdu.dk],
#
# Purpose:    Mosaic data cleaning
#
# Output:     Clean tmp version of the data

# ==== Libraries =====
library(tidyverse)
source("Data_cleaning_scripts/000_Functions.R")
library(foreach)
library(readxl)


# ==== Read data ====
tmp_funky = function(x) x$occhisco %>% unique()
dirs = list.dirs("Data/Raw_data/2509_New_data/Mosaic", full.names = FALSE)
data0 = foreach(d = dirs, .combine = "bind_rows") %do% {
    d_full = paste0("Data/Raw_data/2509_New_data/Mosaic/", d)
    files = list.files(d_full, full.names = TRUE, pattern = "\\_codes.csv$")
    files_labels = list.files(d_full, full.names = TRUE, pattern = "\\_labels.csv$")
    if(length(files)>1) stop("More than one file in folder")
    if(length(files_labels)>1) stop("More than one label file in folder")
    if(length(files)==0) return(NULL) # Skip folders with no data

    data0 = read_csv2(files) %>%
        select(year, relateor, occupat, occhisco) %>%
        mutate(
            source = d
        )

    if(tmp_funky(data0) %>% length() < 10){
        cat("\nLess than 10 unique HISCO codes in", d)
        return(NULL) # Skip folders with no data
    }

    labels = read_csv2(files_labels)

    data0$country = labels$country

    return(data0)
}

# ==== Cleaning data0 ====
# Missing values are marked by e.g. "!!" or "**" in Mosaic. These still carry
# HISCO codes (e.g. 61110), which is not a training signal for the string.
is_missing_marker = function(x) is.na(x) | str_detect(x, "^[[:punct:][:space:]]*$")

data0 = data0 %>%
  filter(!is_missing_marker(occupat))

data1 = data0 %>%
  rename(
    occ1 = occupat,
    hisco_1 = occhisco
  ) %>%
  select(occ1, hisco_1, country, source)

data2 = data0 %>% # Vesion with relateor added in the occ string
    # Only when relateor is given and differs from the occupation
    # (some sources have the same text in both fields)
    filter(
        !is_missing_marker(relateor),
        str_squish(tolower(relateor)) != str_squish(tolower(occupat))
    ) %>%
    mutate(
        occ1 = paste(relateor, occupat, sep = " ")
    ) %>%
    rename(
        hisco_1 = occhisco
    ) %>% 
    select(occ1, hisco_1, country, source)

data1 = bind_rows(data1, data2)

# ==== More cleaning ====
data1 = data1 %>%
  mutate( # Clean string (lower case as all other sources):
    occ1 = occ1 %>% tolower()
  ) %>%
  mutate(
    hisco_1 = as.numeric(hisco_1),
  ) %>%
  mutate(
    hisco_1 = ifelse(is.na(hisco_1), -1, hisco_1)
  )

data1 = data1 %>%
  mutate(
    hisco_2 = " ",
    hisco_3 = " ",
    hisco_4 = " ",
    hisco_5 = " "
  )

# ==== Country to lang ====
# [1] "Albania"   "France"    "Germany"   "Belarus"   "Poland"    "Lithuania"
# [7] "Serbia"    "Turkey"

data1 = data1 %>% 
    mutate(
        lang = case_when(
            country == "Albania" ~ "sq",
            country == "France" ~ "fr",
            country == "Germany" ~ "ge",
            country == "Belarus" ~ "be",
            country == "Poland" ~ "pl",
            country == "Lithuania" ~ "lt",
            country == "Serbia" ~ "sr",
            country == "Turkey" ~ "tr",
            TRUE ~ "unk"
        )
    )

# ==== Cross walk ====
# Key
# Everything with no exact equivalent in standard HISCO is filtered off

cross_walk = read_csv("Data/Raw_data/O-clack/n2h_2.csv")
key = cross_walk %>%
  # Remove anything with a note
  filter(is.na(comments)) %>% 
  filter(napp.eq.hisco == 1) %>% 
  # Only when we are completely sure of agreement
  filter(hisco.code.num == napp.code.num)

key0 = read_csv("Data/Key.csv")

all_data = data1 %>% 
  semi_join(key, by = c("hisco_1"="hisco.code.num"))

# ==== Get combinations by source and language ====
# By language as well, since some sources span several countries
# (e.g. the Lithuanian Jewish census: be, pl, lt), each with its own word for 'and'

set.seed(20)
sources = all_data %>% distinct(source, lang)
all_data = foreach(s = sources$source, l = sources$lang, .combine = "bind_rows") %do% {
  cat("\nGetting combinations for", s, l)
  data_s = all_data %>% filter(source == s, lang == l) %>% mutate_all(as.character)

  # No combinations for sources with unknown language (no single word for 'and')
  if(any(data_s$lang == "unk")){
    cat(" - skipped (unknown language)")
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

# ==== Check against authoritative HISCO list ====
load("Data/Key.Rdata")

key = key %>% select(hisco, code)

# Remove data not in key (erronoeous data somehow)
all_data %>% 
  filter(!hisco_1 %in% key$hisco)

n1 = NROW(all_data)
all_data = all_data %>% 
  filter(hisco_1 %in% key$hisco) %>% 
  filter(hisco_2 %in% key$hisco) %>% 
  filter(hisco_3 %in% key$hisco) %>% 
  filter(hisco_4 %in% key$hisco) %>% 
  filter(hisco_5 %in% key$hisco)

NROW(all_data) - n1 # 0 observations

# Turn into character
all_data = all_data %>% 
  mutate_all(as.character)

# Add code
all_data = all_data %>% 
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
all_data = all_data %>% 
  ungroup() %>% 
  mutate(RowID = 1:n())

# ==== Save ====
save(all_data, file = "Data/Tmp_data/Mosaic_data.Rdata")

