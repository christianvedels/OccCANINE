# Data from HISCO website
# Created:    2023-11-09
# Auhtors:    Christian Vedel [christian-vs@sam.sdu.dk],
#
# Purpose:    This cleans occupational titles from historyofwork.iisg.nl
#             The underlying data is scraped using: 
#                 'Other_scripts/001_Getting_HISCO_data_from_website.py'
#
# Output:     Clean tmp version of the data

# ==== Libraries ====
library(tidyverse)
library(foreach)
source("Data_cleaning_scripts/000_Functions.R")

# ==== Load data ====
# fs = list.files(
#   "Data/Raw_data/HISCO_website/Many_files", 
#   pattern = ".csv", 
#   full.names = TRUE
# )
# data0 = read_csv(fs, id = "file_name")
# save(data0, file = "Data/Tmp_data/HISCO_website_raw.Rdata")
load("Data/Tmp_data/HISCO_website_raw.Rdata")

# ==== Clean data ====
data0 = data0 %>% 
  select(-...1) %>%
  mutate(
    id = strsplit(file_name, "/")
  ) %>%
  rowwise() %>%
  mutate(
    id = unlist(id[length(id)])
  ) %>%
  ungroup() %>%
  mutate(
    id = gsub(".csv", "", id)
  ) %>%
  select(-file_name) %>%
  pivot_wider(
    names_from = Type,
    values_from = Content,
    id_cols = id
  ) %>%
  select(-Contact) %>%
  rename(
    occ1 = `Occupational title`,
    hisco_1 = `Hisco code`
  ) %>%
  mutate(
    url = paste0(
      "https://historyofwork.iisg.nl/detail_hiswi.php?know_id=",
      id,
      "&lang=")
  ) %>%
  mutate(id = as.numeric(id)) %>% 
  arrange(id)

# Remove invalid HISCO codes
data0 = data0 %>% 
  mutate(
    hisco_1 = as.numeric(hisco_1)
  ) %>% 
  mutate(
    valid_hisco = ifelse(hisco_1 %in% hisco::hisco$hisco, hisco_1, NA)
  )

# ==== Non unique HISCO codes ====
data0 %>% 
  distinct(occ1, Language, Country, valid_hisco) %>% 
  group_by(occ1) %>% 
  count() %>% 
  filter(n>1)

# Summary stats
data0 %>% 
  summarise_all(function(x){sum(is.na(x))})

# Sources
data0$Provenance %>% unique()

# Countries 
data0$Country %>% unique()

# Languages
data0$Language %>% unique()

# Unique hisco codes
data0$hisco_1 %>% unique() %>% length()
data0$valid_hisco %>% unique() %>% length()

# Unique data
data0 %>% 
  distinct(occ1, Language, Country, hisco_1) %>% 
  NROW()

# Manual look at some data
data0 %>% filter(Provenance == 97125) %>% select(url) %>% unlist()
data0 %>% filter(Provenance == 94990) %>% select(url) %>% unlist()
data0 %>% filter(Provenance %in% c(1:11)) %>% select(url) %>% unlist()

# ==== Tranlastions is also good data ====
translations = data0 %>% 
  distinct(hisco_1, Translation) %>% 
  rename(
    occ1 = Translation
  ) %>% 
  mutate(
    Source = "Translations",
    Language = "English"
  ) %>% 
  mutate(hisco_2 = " ")

data0 = data0 %>% 
  bind_rows(translations)

# ==== Preliminary data clean ====
data0 = data0 %>% 
  drop_na(valid_hisco) %>% 
  rename(
    Source = Provenance
  ) %>% 
  mutate( # Clean string:
    occ1 = str_replace_all(occ1, "[^[:alnum:] ]", "") %>% tolower()
  ) %>% 
  mutate(
    occ1 = sub_scandi(occ1) # Clean out all non-English characters
  ) %>% 
  mutate(
    hisco_1 = as.numeric(hisco_1)
  ) %>% 
  mutate(
    hisco_1 = ifelse(is.na(hisco_1), -1, hisco_1)
  )

data0 = data0 %>%
  mutate(
    hisco_2 = " ",
    hisco_3 = " ",
    hisco_4 = " ",
    hisco_5 = " "
  )

# ==== Synthetic combinations ("and") ====
data0 = data0 %>% 
  drop_na(Language) %>%  # 2 observations
  filter(Language != "DA") # 1 observations

unique_languages = data0$Language %>% unique()
print(unique_languages)
paste(unique_languages, collapse = "',\n'") %>% cat()

and_in_lang = c( # https://translated-into.com/and
  'French' = 'et',
  'German' = 'und',
  'Dutch' = 'en',
  'Norwegian' = 'og',
  'Catalan' = 'i',
  'Spanish' = 'y',
  'English' = 'and',
  'Swedish' = 'och',
  'Portugese' = 'e',
  'Danish' = 'og',
  'Greek' = "kai"
)

# Combinations
combinations0 = foreach(i = seq(1:length(and_in_lang)), .combine = "bind_rows") %do% {
  lang_i = names(and_in_lang)[i]
  message1 = paste0(
    "<===================================>\n",
    Sys.time(), ":\nMaking combinations from ", lang_i, "\n"
  )
  cat(message1)
  
  # Extract relevant data
  data_i = data0 %>% filter(
    Language == lang_i
  ) %>% select(occ1, Language, hisco_1, hisco_2)
  
  # Run combinations
  comb_i = Combinations(data_i, and = and_in_lang[i]) %>% 
    mutate_all(as.character)
  
  return(comb_i)
}

combinations0 = combinations0 %>% 
  mutate(Source = paste("Combinations", Language))

# Combine it
data1 = data0 %>% 
  mutate_all(as.character) %>% 
  bind_rows(combinations0)

# ==== Check against authoritative HISCO list ====
load("Data/Key.Rdata")

key = key %>% select(hisco, code)

# Remove data not in key (erronoeous data somehow)
data1 %>% 
  filter(!hisco_1 %in% key$hisco)

n1 = NROW(data1)
data1 = data1 %>% 
  filter(hisco_1 %in% key$hisco) %>% 
  filter(hisco_2 %in% key$hisco)

NROW(data1) - n1 # -9801 observations

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
  mutate(RowID = 1:n()) %>% 
  rename(
    lang = Language
  )

# ==== Standardized language names ====
# 'da', 'en', 'nl', 'se', 'no', 'fr', 'ca', 'unk', 'de', 'is', 'unk', 'it'

stop("Finish this bit")
data1 %>% 
  mutate(
    lang = case_when(
      lang == 'French' ~ 'fr',
      'German' = 'und',
      'Dutch' = 'en',
      'Norwegian' = 'og',
      'Catalan' = 'i',
      'Spanish' = 'y',
      'English' = 'and',
      'Swedish' = 'och',
      'Portugese' = 'e',
      'Danish' = 'og',
      'Greek' = "kai"
    )
  )

and_in_lang = c( # https://translated-into.com/and
  
)

# ==== Save ====
save(data1, file = "Data/Tmp_data/Clean_HISCO_website.Rdata")
