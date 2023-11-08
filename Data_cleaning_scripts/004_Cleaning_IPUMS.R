# Cleaning Ipums data data
# Created:    2023-05-23
# Auhtors:    Christian Vedel [christian-vs@sam.sdu.dk],
#
# Purpose:    This script cleans ipums data
#
# Output:     Clean tmp version of the inputted data

# ==== Libraries ====
rm(list = ls())
source("Data_cleaning_scripts/000_Functions.R")
library(tidyverse)
library(foreach)
library(stringi)
library(stringr)
library(tm)
library(ipumsr)
library(fst)

# # ==== Load data ====
ddi = read_ipums_ddi("Data/Raw_data/IPUMS/ipumsi_00002.xml")
all_data = read_ipums_micro(ddi)

# Fix weird labels
fixIt = function(x){
  x = as_factor(x)
  x = as.character(x)
}

tmp_key = ipums_val_labels(all_data$OCCHISCO)

all_data = all_data %>%
  mutate(
    COUNTRY = fixIt(COUNTRY),
    SAMPLE = fixIt(SAMPLE),
    MARST = fixIt(MARST),
    MARSTD = fixIt(MARSTD),
    OCCHISCO = fixIt(OCCHISCO)
  ) %>%
  left_join(tmp_key, by = c("OCCHISCO"="lbl")) %>%
  rename(HISCO = val)

# # Toy data in script development
# set.seed(20)
# all_data = all_data %>% sample_n(10^6)
# write_fst(all_data, "Data/Tmp_data/IPUMS_tmp_small", compress = 0)
# all_data = read_fst("Data/Tmp_data/IPUMS_tmp_small") 
# 
# all_data %>% 
#   group_by(COUNTRY) %>% 
#   count() %>% 
#   ungroup() %>% 
#   mutate(pct = n/sum(n))

# Key
cross_walk = read_csv("Data/Raw_data/O-clack/n2h_2.csv")
key = cross_walk %>%
  # Remove anything with a note
  filter(is.na(comments)) %>% 
  filter(napp.eq.hisco == 1)


key0 = read_csv("Data/Key.csv")

# ==== Extracting counties ====
all_data$COUNTRY %>% unique() %>% sort()
# "Canada"         "Denmark"        "Egypt"          "France"         "Germany"        "Iceland"        "Ireland"        "Netherlands"
# "Norway"         "Sweden"         "United Kingdom" "United States"

# Canada:
CNT = "Canada"
canada = all_data %>% 
  filter(COUNTRY == CNT) %>% 
  filter(OCCSTRNG != "") %>% 
  select(YEAR, OCCSTRNG, HISCO) %>% 
  mutate(
    HISCO = ifelse(
      HISCO == 99999,
      -1,
      HISCO
    )
  )

# Manual look at data
CNT = "Norway"
norway = all_data %>% 
  filter(COUNTRY == CNT) %>% 
  filter(OCCSTRNG != "") %>% 
  select(YEAR, OCCSTRNG, HISCO) %>% 
  mutate(
    HISCO = ifelse(
      HISCO == 99999,
      -1,
      HISCO
    )
  )

CNT = "Germany"
germany = all_data %>% 
  filter(COUNTRY == CNT) %>% 
  filter(OCCSTRNG != "") %>% 
  select(YEAR, OCCSTRNG, HISCO) %>% 
  mutate(
    HISCO = ifelse(
      HISCO == 99999,
      -1,
      HISCO
    )
  )


CNT = "Iceland"
iceland = all_data %>% 
  filter(COUNTRY == CNT) %>% 
  filter(OCCSTRNG != "") %>% 
  select(YEAR, OCCSTRNG, HISCO) %>% 
  mutate(
    HISCO = ifelse(
      HISCO == 99999,
      -1,
      HISCO
    )
  )

CNT = "United Kingdom"
uk = all_data %>% 
  filter(COUNTRY == CNT) %>% 
  filter(OCCSTRNG != "") %>% 
  select(YEAR, OCCSTRNG, HISCO) %>% 
  mutate(
    HISCO = ifelse(
      HISCO == 99999,
      -1,
      HISCO
    )
  )

CNT = "United States"
usa = all_data %>% 
  filter(COUNTRY == CNT) %>% 
  filter(OCCSTRNG != "") %>% 
  select(YEAR, OCCSTRNG, HISCO) %>% 
  mutate(
    HISCO = ifelse(
      HISCO == 99999,
      -1,
      HISCO
    )
  )

rm(all_data)

# ==== Clean it function ====
Clean_it = function(x){
  # Standardizing strings and var names
  x = x %>%
    mutate(
      HISCO = as.character(HISCO)
    ) %>%
    mutate( # Clean string:
      Original = str_replace_all(OCCSTRNG, "[^[:alnum:] ]", "") %>% tolower()
    ) %>%
    rename(
      occ1 = Original,
      hisco_1 = HISCO
    ) %>% 
    filter(occ1 != "") %>% 
    select(YEAR, occ1, hisco_1) %>%
    mutate( # Remove scandi letters
      occ1 = occ1 %>% sub_scandi()
    ) %>% 
    rename(Year = YEAR)
  
  x = x %>% as.data.frame()
  
  # NA padding
  x = x %>%
    mutate(
      hisco_2 = " ",
      hisco_3 = " ",
      hisco_4 = " ",
      hisco_5 = " "
    )
  
  # Check against valid list
  load("Data/Key.Rdata")
  
  key = key %>% select(hisco, code)
  
  # Remove data not in key (erronoeous data somehow)
  x %>% 
    filter(!hisco_1 %in% key$hisco)
  
  n1 = NROW(x)
  x = x %>% 
    filter(hisco_1 %in% key$hisco) %>% 
    filter(hisco_2 %in% key$hisco) %>% 
    filter(hisco_3 %in% key$hisco) %>% 
    filter(hisco_4 %in% key$hisco) %>% 
    filter(hisco_5 %in% key$hisco)
  
  print(NROW(x) - n1)
  
  # Add code
  x = x %>% 
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
  x = x %>% 
    ungroup() %>% 
    mutate(RowID = 1:n())
  
  return(x)
}

# ==== Canada ====
canada = Clean_it(canada)
save(canada, file = "Data/Tmp_data/Clean_CA_IPUMS.Rdata")

# ==== Norway ====
# Standardizing strings and var names
norway = Clean_it(norway)

save(norway, file = "Data/Tmp_data/Clean_NO_IPUMS.Rdata")

# ==== Germany ====
germany = Clean_it(germany)

save(germany, file = "Data/Tmp_data/Clean_DE_IPUMS.Rdata")

# ==== Iceland ====
# Standardizing strings and var names
iceland = Clean_it(iceland)

# Remove bbbbb
iceland = iceland %>% 
  filter(occ1 != "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")

# Remove numbers
iceland = iceland %>% 
  filter(is.na(as.numeric(occ1)))
  
save(iceland, file = "Data/Tmp_data/Clean_IS_IPUMS.Rdata")

# ==== UK ====
uk = Clean_it(uk)
uk
save(uk, file = "Data/Tmp_data/Clean_UK_IPUMS.Rdata")

# ==== USA ====
usa = Clean_it(usa)
usa
save(usa, file = "Data/Tmp_data/Clean_USA_IPUMS.Rdata")

