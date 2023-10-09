# Cleaning CEDAR
# Created:    2023-10-09
# Auhtors:    Christian Vedel [christian-vs@sam.sdu.dk],
#
# Purpose:    This script cleans Swedish census data SWEPOP
#
# Output:     Clean tmp version of the inputted data

# ==== Libraries =====
library(tidyverse)
source("Data_cleaning_scripts/000_Functions.R")
library(foreach)

# ==== Read data ====
data0 = read_csv2("Data/Raw_data/SWEDPOP_data/individual.csv", locale = locale(encoding = "ISO-8859-1"))

for(f in list.files("Data/Raw_data/SWEDPOP_data")){
  if(grepl(".csv", f)){
    x = read_csv2(paste0("Data/Raw_data/SWEDPOP_data/", f), locale = locale(encoding = "ISO-8859-1"), n_max = 1000)
    print(f)
    print(colnames(x))
  }
}


# ==== Cleaning data0 ====
data0 = data0 %>%
  rename(
    occ1 = OCCUPATION,
    hisco_1 = HISCO
  ) %>% 
  mutate( # Clean string:
    occ1 = str_replace_all(occ1, "[^[:alnum:] ]", "") %>% tolower()
  ) %>% 
  mutate(
    occ1 = sub_scandi(occ1)
  ) %>% 
  select(occ1, hisco_1) %>% 
  mutate(
    hisco_1 = as.numeric(hisco_1)
  )

data0 = data0 %>%
  mutate(
    hisco_2 = " ",
    hisco_3 = " ",
    hisco_4 = " ",
    hisco_5 = " "
  )

# Add extra occupations with 'och'
set.seed(20)
# This is a growing vector, and it is slow. But this was quick to implement.
# Please forgive me
combinations = foreach(i = 1:NROW(data0), .combine = "bind_rows") %do% {
  occ_i = data0$occ1[i]
  
  if(data0$hisco_1[i] == -1){ # Handle no occupation
    return(
      data.frame(occ1 = NA)
    )
  }
  
  # Generate 10 combinations with 'and' (och)
  x10_combinations = foreach(r = 1:10, .combine = "bind_rows") %do% {
    occ_r = sample(data0$occ1, 1)
    occ_ir = paste(occ_i, "och", occ_r)
    hisco_r = data0 %>% 
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
          hisco_1 = data0$hisco_1[i],
          hisco_2 = " "
        ) %>% mutate_all(as.character)
      )
    }
    
    data.frame(
      occ1 = occ_ir,
      hisco_1 = data0$hisco_1[i],
      hisco_2 = hisco_r
    ) %>% remove_rownames() %>% mutate_all(as.character)
  }
  
  res = x10_combinations %>% mutate_all(as.character)
  
  cat(i, "of", NROW(data0), "             \r")
  
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

NROW(data1) - n1 # 107 observations

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
save(data1, file = "Data/Tmp_data/")
