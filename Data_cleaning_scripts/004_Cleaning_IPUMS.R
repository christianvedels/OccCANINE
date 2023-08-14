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

# # ==== Load data ====
ddi = read_ipums_ddi("Data/Raw_data/ipumsi_00002.xml")
all_data = read_ipums_micro(ddi)

# Toy data in script development
set.seed(20)
all_data = all_data %>% sample_n(10^6)

all_data %>% 
  group_by(COUNTRY) %>% 
  count() %>% 
  ungroup() %>% 
  mutate(pct = n/sum(n))

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

# ==== List countries ====
all_data$COUNTRY %>% unique() %>% sort()
# "Canada"         "Denmark"        "Egypt"          "France"         "Germany"        "Iceland"        "Ireland"        "Netherlands"
# "Norway"         "Sweden"         "United Kingdom" "United States"


all_data = all_data %>% 
  filter(COUNTRY != "Denmark") %>% # We already have higher quality DK data
  filter(COUNTRY != "Egypt") %>% # No raw occupational descriptions in this data
  filter(COUNTRY != "France") %>% # No raw occupational descriptions in this data
  filter(COUNTRY != "Ireland") %>% # No raw occupational descriptions in this data
  filter(COUNTRY != "Netherlands") %>% # No raw occupational descriptions in this data
  ungroup()

# Canada:
# Contains both French and English
# Problem with BBBBBBBB in Icelandic data
# Norway is missing household position variable, where a lot of information is

# Manual look at data
CNT = "Norway"
all_data %>% 
  filter(COUNTRY == CNT) %>% View()

all_data %>% 
  filter(COUNTRY == CNT) %>% 
  group_by(OCCSTRNG == "") %>% 
  count()

# Fix inconsistency in "no occupation" encoding
all_data %>% 
  mutate(
    HISCO = ifelse(HISCO == 99999, -1, HISCO)
  )



# # # ==== Data cleaning ====
# Fixing norway
stop("Fix Norway")

# Standardizing strings and var names
NROW(all_data)
all_data = all_data %>%
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
  filter(occ1 != "")

all_data = all_data %>% as.data.frame()

# NA padding
all_data = all_data %>%
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
text = sample(all_data$occ1, 10000)
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

# # ==== Encode with key ====
load("Data/Key.Rdata")

key = key %>% select(hisco, code)

# 00-Pruning categories
#     Many newly custom categories contained in IPUMS
#     Some of them can be converted back to standard
#     or something close to it, by inserting 00 as the
#     last two digits
# Find missing
# missing_hiscos = all_data %>% 
#   count(hisco_1, OCCHISCO) %>% 
#   arrange(hisco_1) %>% 
#   left_join(key, by = c("hisco_1"="hisco")) %>% 
#   filter(is.na(code)) %>% 
#   arrange(-n)
# 
# missing_hiscos = missing_hiscos %>% 
#   rowwise() %>% 
#   mutate(
#     hisco_replace = sub("\\d{3}$", "000", hisco_1)
#   ) %>% 
#   select(hisco_1, hisco_replace)
# 
# all_data = all_data %>% 
#   left_join(missing_hiscos, by = c("hisco_1")) %>% 
#   mutate(
#     hisco_1 = ifelse(
#       !is.na(hisco_replace),
#       hisco_replace,
#       hisco_1
#     )
#   ) %>% 
#   select(-hisco_replace)

# Remove data not in key (erronoeous data somehow)
n1 = NROW(all_data)
occ_data = all_data %>%
  filter(hisco_1 %in% key$hisco)

NROW(all_data) - n1 # -72337 of a million

# Add code
all_data = all_data %>%
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

all_data = all_data %>%
  mutate(
    COUNTRY = gsub(" ", "_", COUNTRY)
  ) %>% 
  ungroup() %>%
  mutate(RowID = paste0(COUNTRY, YEAR, 1:n()))

# ==== Save data ====
# Save in loop
for(n in unique(all_data$COUNTRY)){
  # Construct fname
  fname_n = paste0("Data/Tmp_data/Clean_IPUMS_",n,".Rdata")
  
  # Filter data and save
  all_data_n = all_data %>% 
    filter(COUNTRY == n)
  save(all_data_n, file = fname_n)
  
  cat("Saved:",fname_n,"\n")
}
