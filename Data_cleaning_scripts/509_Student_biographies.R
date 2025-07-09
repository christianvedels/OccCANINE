# Cleaning data from Student biographies (2025)
# Updated:    2025-07-09
# Auhtors:    Christian Vedel [christian-vs@sam.sdu.dk],
#
# Purpose:    Cleans the data to get it ready to test performance

# ==== Libraries ====
library(tidyverse)
library(readxl)
source("Data_cleaning_scripts/000_Functions.R")

# ==== Read data ====
data0 = read_excel(
    "Data/Raw_data/2412_New_data/Student_biographies/Kopi av Biographical_data_studentårbøker Norge_1831_sendt_til_Christian.xlsx",
    skip = 2
)

# ==== Cleaning data0 ====
data1 = data0 %>%
  rename(
    occ1 =  `Arbeid far`
  ) %>%
  select(occ1) %>%
  drop_na() %>%
  mutate(
    RowID = row_number()
  )

# Take sample of 1000 rows
set.seed(20)
data1 = data1 %>%
  sample_n(500) %>%
  drop_na()


# ==== Save ====
data1 %>% write_csv0("Data/OOD_data/NO_student_biographies.csv")
