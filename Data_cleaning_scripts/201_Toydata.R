# Toydata to show the model
# Updated:    2024-02-19
# Auhtors:    Christian Vedel [christian-vs@sam.sdu.dk],
#
# Output:     Train, test and validation data

library(tidyverse)
set.seed(20)
read_csv("Data/Validation_data/EN_marr_cert_val.csv") %>% 
  filter(nchar(occ1)>15) %>% 
  sample_n(10000) %>% 
  select(occ1, hisco_1) %>% 
  write_csv("OccCANINE/Data/TOYDATA.csv")
