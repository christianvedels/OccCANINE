# Cleaning Historical Sample of the Netherlands data
# Created:    2023-08-13
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

# # ==== Load data ====
all_data = read_csv("Data/Raw_data/HSN_HISCO_release_2018_01a.csv")

# Toy data in script development
# set.seed(20)
# all_data = all_data %>% sample_frac(0.01)

# # ==== Data cleaning 1 ====
occ_data = all_data %>%
  mutate(
    HISCO = as.character(HISCO)
  ) %>% 
  mutate( # Clean string:
    Original = str_replace_all(Original, "[^[:alnum:] ]", "") %>% tolower()
  ) %>% 
  rename(
    occ1 = Original,
    hisco_1 = HISCO
  )

occ_data = occ_data %>% as.data.frame()

# NA padding
occ_data = occ_data %>%
  mutate(
    hisco_2 = " ",
    hisco_3 = " ",
    hisco_4 = " ",
    hisco_5 = " "
  )

# ==== Some simple descriptive stats ====
# Word cloud
library(wordcloud)
set.seed(20)
text = sample(occ_data$occ1, 10000)
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
n1 = NROW(occ_data)
occ_data = occ_data %>% 
  filter(hisco_1 %in% key$hisco) %>% 
  filter(hisco_2 %in% key$hisco) %>% 
  filter(hisco_3 %in% key$hisco) %>% 
  filter(hisco_4 %in% key$hisco) %>% 
  filter(hisco_5 %in% key$hisco)

NROW(occ_data) - n1 # 7 observations

# Add code
occ_data = occ_data %>% 
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
save(occ_data, file = "Data/Tmp_data/Clean_HSN_database.Rdata")
