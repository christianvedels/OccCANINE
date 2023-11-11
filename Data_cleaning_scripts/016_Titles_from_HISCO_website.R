# Data from HISCO website
# Created:    2023-11-09
# Auhtors:    Christian Vedel [christian-vs@sam.sdu.dk],
#
# Purpose:    This script scrapes and cleans occupational titles from historyofwork.iisg.nl
#
# Output:     Clean tmp version of the data

# ==== Libraries ====
library(tidyverse)
library(foreach)

# ==== Load data ====
fs = list.files(
  "Data/Raw_data/HISCO_website/Many_files", 
  pattern = ".csv", 
  full.names = TRUE
)
data0 = read_csv(fs, id = "file_name")

# ==== Clean data ====
data0 = data0 %>% 
  select(-...1) %>%
  mutate(
    id = strsplit(file_name, "/")
  ) %>%
  rowwise() %>%
  mutate(
    id = unlist(id[length(id)])
  ) %>%
  ungroup() %>%
  mutate(
    id = gsub(".csv", "", id)
  ) %>%
  select(-file_name) %>%
  pivot_wider(
    names_from = Type,
    values_from = Content,
    id_cols = id
  ) %>%
  select(-Contact) %>%
  rename(
    occ1 = `Occupational title`,
    hisco_1 = `Hisco code`
  ) %>%
  mutate(
    url = paste0(
      "https://historyofwork.iisg.nl/detail_hiswi.php?know_id=",
      id,
      "&lang=")
  ) %>%
  mutate(id = as.numeric(id)) %>% 
  arrange(id)

# ==== Non unique HISCO codes ====
data0 %>% 
  distinct(occ1, Language, Country, hisco_1) %>% 
  group_by(occ1) %>% 
  count() %>% 
  filter(n>1)

# Summary stats
data0 %>% 
  summarise_all(function(x){sum(is.na(x))})

# Sources
data0$Provenance %>% unique()

# Countries 
data0$Country %>% unique()

# Languages
data0$Language %>% unique()

# Unique hisco codes
data0$hisco_1 %>% unique() %>% length()

# Unique data
data0 %>% 
  distinct(occ1, Language, Country, hisco_1) %>% 
  NROW()



# ==== To rescrape ====
pt1 = data0 %>% 
  filter(is.na(occ1)) %>% 
  select(id)

# Running numbers
skipping_ids = data0 %>% 
  mutate(dif = as.numeric(id) - dplyr::lag(as.numeric(id))) %>% 
  filter(dif != 1) %>% select(id, dif)

skipped_ids = foreach(i = 1:NROW(skipping_ids)) %do% {
  skipping_ids$id[i] - seq(1, skipping_ids$dif[i]-1)
} %>% unlist()

skipped_ids %>% length()

# %>% 
#   write_csv2("Data/Raw_data/HISCO_website/To_rescrape")
# 

