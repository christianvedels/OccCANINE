# Cleaning BHMD (Barcelona Historical Mariage Database)
# Created:    2023-11-06
# Auhtors:    Christian Vedel [christian-vs@sam.sdu.dk],
#
# Purpose:    This script cleans Swedish data from Barcelona Historical Mariage Database
#
# Output:     Clean tmp version of the inputted data

# ==== Libraries =====
library(tidyverse)
source("Data_cleaning_scripts/000_Functions.R")
library(foreach)
library(readxl)

# ==== Read data ====
data0 = read_excel("Data/Raw_data/Barcelona Historical Marriage Database/Occupation Matrix BHMD_3001.xlsx")

# ==== Cleaning data0 ====
# Raw
data0 = data0 %>%
  rowwise() %>%
  mutate(occ_std = ifelse(
    is.na(`Standardized Occupational Title 2`),
    `Standardized Occupational Title 1`,
    paste0(c(
      NA_str(`Standardized Occupational Title 1`),
      NA_str(`Standardized Occupational Title 2`)
    ), collapse = " y ")
  )) %>% 
  select(-`Standardized Occupational Title 1`, -`Standardized Occupational Title 2`) %>% 
  ungroup() %>% 
  rename(occ_norm = `Normalized Occupational Title`) %>% 
  rename(occ_raw = `Raw Ocupational Title`) %>% 
  rename(
    hisco_1 = `Hisco Code_1`,
    hisco_2 = `Hisco Code_2`,
    hisco_3 = `Hisco Code_3`
  ) %>% 
  pivot_longer(
    cols = c(occ_raw, occ_norm, occ_std),
    names_to = "Type",
    values_to = "occ1"
  ) %>% 
  select(Type, occ1, hisco_1:hisco_3) %>% 
  mutate( # Clean string:
    occ1 = str_replace_all(occ1, "[^[:alnum:] ]", "") %>% tolower()
  ) %>% 
  mutate(
    occ1 = sub_scandi(occ1)
  ) %>% 
  mutate(
    hisco_1 = NA_str(hisco_1),
    hisco_2 = NA_str(hisco_2),
    hisco_3 = NA_str(hisco_3),
  )

data0 = data0 %>%
  mutate(
    hisco_4 = " ",
    hisco_5 = " "
  ) %>%
  drop_na(hisco_1)

# Add extra occupations with 'och'
set.seed(20)
# This is a growing vector, and it is slow. But this was quick to implement.
# Please forgive me
data00 = data0 %>% filter(hisco_2 == " ")

combinations = foreach(i = 1:NROW(data00), .combine = "bind_rows") %do% {
  occ_i = data00$occ1[i]
  
  if(data00$hisco_1[i] == -1){ # Handle no occupation
    return(
      data.frame(occ1 = NA)
    )
  }
  
  # Generate 10 combinations with 'and' (och)
  x10_combinations = foreach(r = 1:10, .combine = "bind_rows") %do% {
    occ_r = sample(data00$occ1, 1)
    occ_ir = paste(occ_i, "y", occ_r)
    hisco_r = data00 %>% 
      filter(occ1 == occ_r) %>% 
      select(hisco_1) %>% unlist()
    
    if(length(hisco_r)>1){
      hisco_r = hisco_r[1]
    }
    
    # If no occupaiton
    if(hisco_r == -1){
      return(
        data.frame(
          occ1 = occ_ir,
          hisco_1 = data00$hisco_1[i],
          hisco_2 = " "
        ) %>% mutate_all(as.character)
      )
    }
    
    data.frame(
      occ1 = occ_ir,
      hisco_1 = data00$hisco_1[i],
      hisco_2 = hisco_r
    ) %>% remove_rownames() %>% mutate_all(as.character)
  }
  
  res = x10_combinations %>% mutate_all(as.character)
  
  cat(i, "of", NROW(data00), "             \r")
  
  return(res)
}


combinations = combinations %>%
  mutate(
    hisco_3 = " ",
    hisco_4 = " ",
    hisco_5 = " "
  )

data1 = data0 %>% 
  mutate_all(as.character) %>% 
  bind_rows(combinations)

data1 = data1 %>% 
  drop_na(occ1)

# Fix no occ
funky = function(x){
  ifelse(
    x %in% c(-101:(-106)),
    -1,
    x
  )
}
data1 = data1 %>% 
  mutate(
    hisco_1 = funky(hisco_1),
    hisco_2 = funky(hisco_2),
    hisco_3 = funky(hisco_3),
    hisco_4 = funky(hisco_4),
    hisco_5 = funky(hisco_5)
  )

# ==== Check against authoritative HISCO list ====
load("Data/Key.Rdata")

key = key %>% select(hisco, code)

# Remove data not in key (erronoeous data somehow)
data1 %>% 
  filter(!hisco_1 %in% key$hisco)

n1 = NROW(data1)
data1 = data1 %>% 
  filter(hisco_1 %in% key$hisco) %>% 
  filter(hisco_2 %in% key$hisco) %>% 
  filter(hisco_3 %in% key$hisco) %>% 
  filter(hisco_4 %in% key$hisco) %>% 
  filter(hisco_5 %in% key$hisco)

NROW(data1) - n1 # 21405 observations

# Turn into character
data1 = data1 %>% 
  mutate_all(as.character)

# Add code
data1 = data1 %>% 
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
data1 = data1 %>% 
  ungroup() %>% 
  mutate(RowID = 1:n())

# ==== Save ====
save(data1, file = "Data/Tmp_data/Clean_CA_BCN.Rdata")
