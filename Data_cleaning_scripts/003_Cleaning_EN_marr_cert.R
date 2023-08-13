# Cleaning En marriage certificate data
# Created:    2023-05-23
# Authors:    Christian Vedel [christian-vs@sam.sdu.dk],
#
# Purpose:    This script cleans English marriage certificate data
#
# Output:     Clean tmp version of the inputted data

# ==== Libraries ====
rm(list = ls())
source("Data_cleaning_scripts/000_Functions.R")
library(tidyverse)
library(foreach)

# ==== Load data ====
all_data = read.csv("Data/Raw_data/EN_MARIAGE_CERTIFICATES.csv")

# Select relevant vars 
all_data = all_data %>% select(occraw, hisco, n)

# ==== Expand data ====
# foreach(i = 1:NROW(all_data)) %do% {
#   fname_i = paste0("Data/Tmp_data/", i, ".Rdata")
# 
#   if(paste0(i,".Rdata") %in% list.files("Data/Tmp_data")){
#     cat("Skipped", i, "                \r")
#     return(NA)
#   }
# 
#   data_i = all_data[i,]
#   data_i = data.frame(
#     hisco = rep(data_i$hisco, data_i$n),
#     occraw = data_i$occraw,
#     n = data_i$n
#   )
# 
#   save(data_i, file = fname_i)
# 
#   cat(i,"of",NROW(all_data),"        \r")
# }
# 
# data0 = foreach(i = 1:NROW(all_data)) %do% {
#   fname_i =  paste0("Data/Tmp_data/", i, ".Rdata")
#   load(fname_i)
#   cat(i,"of",NROW(all_data),"        \r")
#   return(data_i)
# }
# 
# foreach(i = 1:NROW(all_data)) %do% {
#   fname_i =  paste0("Data/Tmp_data/", i, ".Rdata")
#   cat(i,"of",NROW(all_data),"        \r")
#   unlink(fname_i)
# }
# 
# save(data0, file = "Data/Tmp_data/Tmp_EN_marr1.Rdata")
load("Data/Tmp_data/Tmp_EN_marr1.Rdata")

# Bind frames together
data0 = do.call("rbind", data0)

# Shuffle rows
set.seed(20)
data0 = data0 %>% sample_frac(1)

# Clean strings
data0 = data0 %>% 
  rename(
    occ1 = occraw
  ) %>% 
  mutate(
    occ1 = tolower(occ1)
  ) %>% 
  mutate(
    occ1 = gsub("&", "and", occ1)
  )

# ==== Verified unlabelled data ====
# The original labels in this data is from a regex procedure. A central concern
# is whether the observations marked '-1' really do not have an occupation or
# or whether they were just not caught in the regex. 
# 20,000 samples of '-1' data is extracted and verified to not have an occupation.
# All the rest are thrown out of the sample
# 
# set.seed(20)
# no_occ = data0 %>%
#   filter(is.na(hisco)) %>%
#   sample_n(20000) %>%
#   group_by(occ1) %>%
#   count() %>%
#   arrange(-n) %>%
#   ungroup() %>%
#   mutate(pct = cumsum(n)/sum(n))
# 
# # no_occ %>% filter(pct < 0.95) %>% NROW()
# 
# no_occ %>%
#   write_csv2("Data/Manual_data/Verified_no_occupation_UK_marr_cert.csv")

# Load verified no occupation
no_occ = readxl::read_excel("Data/Manual_data/Verified_no_occupation_UK_marr_cert.xlsx")

# Select relevant
no_occ = no_occ %>% 
  filter(verified == 1) %>% 
  select(occ1, n) %>% 
  mutate(hisco = "-1")

# Expand
no_occ = foreach(i = 1:NROW(no_occ), .combine = "bind_rows") %do% {
  data.frame(
    occ1 = rep(no_occ$occ1[i], no_occ$n[i]),
    hisco = -1
  )
}

set.seed(20)
no_occ = no_occ %>% sample_frac(1)

# Merge
set.seed(20)
data1 = data0 %>% 
  filter(!is.na(hisco)) %>% 
  bind_rows(no_occ) %>% 
  sample_frac(1)

# Add empty vars for format til conform
data1 = data1 %>% 
  rename(
    hisco_1 = hisco
  ) %>% 
  mutate(
    hisco_2 = " ",
    hisco_3 = " ",
    hisco_4 = " ",
    hisco_5 = " ",
  ) %>% 
  ungroup() %>% 
  mutate(
    RowID = 1:n()
  ) %>% 
  select(-n) %>% 
  mutate(hisco_1 = as.character(hisco_1))

# ==== Encode with key ====
load("Data/Key.Rdata")

key = key %>% select(hisco, code)

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

# ==== Save data ====
save(data1, file = "Data/Tmp_data/Clean_EN_marr_cert.Rdata")
