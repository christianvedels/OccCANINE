# Cleaning DK census data
# Created:    2023-05-23
# Authors:    Christian Vedel [christian-vs@sam.sdu.dk],
#
# Purpose:    This script cleans Danish census data
#
# Output:     Clean tmp version of the inputted data

# ==== Libraries ====
rm(list = ls())
source("Data_cleaning_scripts/000_Functions.R")
library(tidyverse)
library(foreach)
library(stringi)
library(tm)

# # ==== Load data ====
# load("Data/Raw_data/DK_CENSUS_DATA.Rdata")
# all_data = merged_data
# rm(merged_data)
# 
# # Toy data in script development
# # set.seed(20)
# # all_data = all_data %>% sample_frac(0.01)
# 
# # ==== Data cleaning 1 ====
# occ_data = all_data %>%
#   mutate(
#     hisco_1 = as.character(hisco_1),
#     hisco_2 = as.character(hisco_2),
#     hisco_3 = as.character(hisco_3),
#     hisco_4 = as.character(hisco_4),
#     hisco_5 = as.character(hisco_5)
#   ) %>%
#   mutate( # If the HISCO code has a leading zero it is read incorrectly
#     hisco_1 = ifelse(nchar(hisco_1) == 4, paste(sep = '', 0,hisco_1), hisco_1),
#     hisco_2 = ifelse(nchar(hisco_2) == 4, paste(sep = '', 0,hisco_2), hisco_2),
#     hisco_3 = ifelse(nchar(hisco_3) == 4, paste(sep = '', 0,hisco_3), hisco_3),
#     hisco_4 = ifelse(nchar(hisco_4) == 4, paste(sep = '', 0,hisco_4), hisco_4),
#     hisco_5 = ifelse(nchar(hisco_5) == 4, paste(sep = '', 0,hisco_5), hisco_5)
#   ) %>%
#   mutate( # Occ string which also contains Household status
#     new_occ_string = paste(Occupation, Household_status)
#   ) %>%
#   mutate(
#     new_occ_string = ifelse( # In this case the string should not be repeated
#       Occupation == Household_status,
#       Occupation,
#       new_occ_string
#     )
#   )
# 
# occ_data = occ_data %>% as.data.frame()
# 
# # Select relevant data
# occ_data = occ_data %>%
#   rename(
#     occ1 = new_occ_string
#   ) %>%
#   select(Year, RowID, occ1, hisco_1:hisco_5, Household_status, Occupation)
# 
# # Save this processing stage
# save(occ_data, file = "Data/Tmp_data/Tmp_dk_census1.Rdata")
# load("Data/Tmp_data/Tmp_dk_census1.Rdata")
# 
# # ==== Data cleaning 2 ====
# # tmp_funky1: Makes NAs -1
# tmp_funky1 = function(x) ifelse(is.na(x)|x == "", -1, x)
# 
# occ_data0 = occ_data
# occ_data = occ_data0 %>%
#   mutate(
#     labelled = hisco_1 != ""
#   ) %>%
#   mutate(
#     labelled = ifelse(is.na(hisco_1), FALSE, labelled)
#   ) %>%
#   mutate(
#     Occupation = ifelse(is.na(Occupation), "", Occupation),
#     Household_status = ifelse(is.na(Household_status), "", Household_status),
#     occ1 = ifelse(is.na(occ1), "", occ1)
#   ) %>%
#   mutate( # Remove scandi letters
#     occ1 = occ1 %>% sub_scandi()
#   ) %>%
#   mutate( # To lower
#     occ1 = tolower(occ1)
#   ) %>%
#   mutate(
#     hisco_1 = tmp_funky1(hisco_1),
#     hisco_2 = tmp_funky1(hisco_2),
#     hisco_3 = tmp_funky1(hisco_3),
#     hisco_4 = tmp_funky1(hisco_4),
#     hisco_5 = tmp_funky1(hisco_5)
#   ) %>%
#   mutate(
#     occ1 = gsub('[[:punct:] ]+',' ', occ1)
#   ) %>%
#   mutate(
#     occ1 = trimws(occ1)
#   ) %>%
#   mutate(
#     occ1 = ifelse(occ1 == "", " ", occ1)
#   )
# 
# # Fix recurring codes and shuffle them
# shuffle_codes = function(x){ # Shuffle codes but keeps na-codes to the right
#   non_na = x[which(!x %in% c(-1))]
#   nas = x[which(x %in% c(-1))]
#   shuffled_x = sample(non_na, replace = FALSE)
#   res = c(shuffled_x, nas)
#   if(!length(res)==5) stop("This should not happen")
#   return(res)
# }
# 
# tmp220903 = function(x){
#   x = x %>%
#     unlist() %>%
#     unique() %>%
#     sort() %>%
#     rev()
# 
#   # pad
#   res = rep("-1", 5)
#   for(j in 1:length(x)){
#     res[j] = x[j]
#   }
# 
#   res = shuffle_codes(res)
# 
#   return(res)
# }
# 
# occ_data =
#   occ_data %>%
#   rowwise() %>%
#   mutate(
#     tmp = list(c(hisco_1, hisco_2, hisco_3, hisco_4, hisco_5))
#   ) %>%
#   mutate(
#     tmp = list(tmp220903(tmp))
#   ) %>%
#   mutate(
#     hisco_1 = tmp[1],
#     hisco_2 = tmp[2],
#     hisco_3 = tmp[3],
#     hisco_4 = tmp[4],
#     hisco_5 = tmp[5]
#   ) %>% ungroup() %>%
#   select(-tmp)
# 
# save(occ_data, file = "Data/Tmp_data/Tmp_dk_census2.Rdata")
# load("Data/Tmp_data/Tmp_dk_census2.Rdata")
# 
# # ==== Labelled unlabelled data ====
# # Years with labeling
# labelled_years = occ_data %>% 
#   group_by(Year) %>%
#   summarise(
#     na_label_pct = sum(hisco_1 == "-1")/n()
#   ) %>% 
#   filter(
#     na_label_pct<1
#   ) %>% 
#   select(Year) %>% 
#   unlist()
# 
# # 1787, 1834, 1845 and 1880 has labels
# occ_data = occ_data %>% 
#   mutate(
#     labelled = ifelse(Year %in% labelled_years, 1, 0)
#   )

# ==== Verified unlabelled data ====
# The original labels in this data is from a regex procedure. A central concern
# is whether the observations marked '-1' really do not have an occupation or
# or whether they were just not caught in the regex. 
# 20,000 samples of '-1' data is extracted and verified to not have an occupation.
# All the rest are thrown out of the sample
# 
# set.seed(20)
# no_occ = occ_data %>%
#   filter(hisco_1 == "-1") %>%
#   filter(labelled == 1) %>%
#   sample_n(20000) %>%
#   group_by(occ1) %>%
#   count() %>%
#   arrange(-n) %>% 
#   ungroup() %>% 
#   mutate(pct = cumsum(n)/sum(n))
# 
# no_occ %>% filter(pct < 0.95) %>% NROW()
# 
# no_occ %>%
#   write_csv2("Data/Manual_data/Verified_no_occupation_DK_census.csv")

# Load verified no occupation
no_occ = readxl::read_excel("Data/Manual_data/Verified_no_occupation_DK_census.xlsx")

# Misc clean
no_occ = no_occ %>% 
  filter(verified == 1) %>% 
  select(occ1, n) %>% 
  mutate(occ1 = ifelse(is.na(occ1), " ", occ1))

# All data which match this description
occ_data_verified_no_occ = occ_data %>% 
  filter(labelled == 1) %>% 
  semi_join(no_occ, by = "occ1") %>% 
  mutate(
    hisco_1 = "-1", 
    hisco_2 = "-1", 
    hisco_3 = "-1",
    hisco_4 = "-1",
    hisco_5 = "-1"
  )

# Data with HISCO codes
occ_data_with_HISCOs = occ_data %>% 
  anti_join(no_occ, by = "occ1") %>% 
  filter(hisco_1 != "-1") %>% 
  filter(labelled == 1)

# Merge the two
set.seed(20)
occ_data_labelled = occ_data_verified_no_occ %>% 
  bind_rows(occ_data_with_HISCOs) %>% 
  sample_frac(1) # Shuffle rows

# Make hisco codes numeric
occ_data_labelled = occ_data_labelled %>% 
  mutate(
    hisco_1 = as.numeric(hisco_1),
    hisco_2 = as.numeric(hisco_2),
    hisco_3 = as.numeric(hisco_3),
    hisco_4 = as.numeric(hisco_4),
    hisco_5 = as.numeric(hisco_5)
  )

# NA padding
occ_data_labelled = occ_data_labelled %>%
  mutate(
    hisco_1 = ifelse(
      hisco_1 == -1 &
        hisco_2 == -1 &
        hisco_3 == -1 &
        hisco_4 == -1 &
        hisco_5 == -1,
      -1,
      ifelse(hisco_1 == -1, " ", hisco_1)
    ),
    hisco_2 = ifelse(hisco_2 == -1, " ", hisco_2),
    hisco_3 = ifelse(hisco_3 == -1, " ", hisco_3),
    hisco_4 = ifelse(hisco_4 == -1, " ", hisco_4),
    hisco_5 = ifelse(hisco_5 == -1, " ", hisco_5)
  )
  
# ==== Some simple descriptive stats ====
# Word cloud
library(wordcloud)
set.seed(20)
text = sample(occ_data_labelled$occ1, 10000)
docs = Corpus(VectorSource(text))
dtm = TermDocumentMatrix(docs)
the_mat = as.matrix(dtm)
words = sort(rowSums(the_mat),decreasing=TRUE)
df = data.frame(word = names(words), freq = words)

wordcloud(
  words = df$word, 
  freq = df$freq, 
  min.freq = 1, 
  max.words=200, 
  random.order=FALSE, 
  rot.per=0.35, 
  colors=brewer.pal(8, "Dark2")
)

# ==== Encode with key ====
load("Data/Key.Rdata")

key = key %>% select(hisco, code)

# Remove data not in key (erronoeous data somehow)
n1 = NROW(occ_data_labelled)
occ_data_labelled = occ_data_labelled %>% 
  filter(hisco_1 %in% key$hisco) %>% 
  filter(hisco_2 %in% key$hisco) %>% 
  filter(hisco_3 %in% key$hisco) %>% 
  filter(hisco_4 %in% key$hisco) %>% 
  filter(hisco_5 %in% key$hisco)

NROW(occ_data_labelled) - n1 # Removed 187 observations

# Add code
occ_data_labelled = occ_data_labelled %>% 
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
save(occ_data_labelled, file = "Data/Tmp_data/Clean_DK_census.Rdata")
