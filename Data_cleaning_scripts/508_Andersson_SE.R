# Cleaning data from Anderson (2025)
# Updated:    2025-07-09
# Auhtors:    Christian Vedel [christian-vs@sam.sdu.dk],
#
# Purpose:    Cleans the data to get it ready to test performance

# ==== Libraries ====
library(tidyverse)
library(readxl)
source("Data_cleaning_scripts/000_Functions.R")

# ==== Read data ====
data0 = read_excel("Data/Raw_data/2412_New_data/Anderson_data/data/hisco_predict.xlsx")

# ==== Cleaning data0 ====
data1 = data0 %>%
  rename(
    occ1 = occstrng
  ) %>%
  select(occ1) %>%
  mutate(
    RowID = row_number()
  )


# ==== Save ====
data1 %>% write_csv0("Data/OOD_data/SE_andersson.csv")