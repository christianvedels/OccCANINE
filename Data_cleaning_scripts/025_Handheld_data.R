# Handheld data cleaning
# Created:    2025-09-24
# Updated:    2026-10-17
# Auhtors:    Christian Vedel [christian-vs@sam.sdu.dk],
#
# Purpose:    This script cleans Handheld data (data that we produce ourselves)
#
# Output:     Clean tmp version of the data

# ==== Libraries =====
library(tidyverse)
library(foreach)

# ==== Read data ====
fisherman_and_farmers = read_csv("Data/Raw_data/2509_New_data/Generated_manual/Farmer_and_fisherman/farmer_fisherman_translations.csv")

# Read all content in the relevant folder
fisherman_and_farmer_variations = list.files("Data/Raw_data/2509_New_data/Generated_manual/Farmer_and_fisherman/by_lang") %>%
    map_df(function(x){
        f = file.path("Data/Raw_data/2509_New_data/Generated_manual/Farmer_and_fisherman/by_lang", x)
        df = read_csv(f, col_names = "occ1", skip = 1) # Overwrite column names
        df$lang = x
        return(df)
    }) %>%
    mutate(lang = gsub(".csv", "", lang))

no_occupation =  list.files("Data/Raw_data/2509_New_data/Generated_manual/No_occupation") %>%
    map_df(function(x){
        f = file.path("Data/Raw_data/2509_New_data/Generated_manual/No_occupation", x)
        df = read_csv(f, col_names = c("occ1", "category"), skip = 1, col_types = "cc") # Overwrite column names
        df$lang = x
        return(df)
    }) %>%
    mutate(lang = gsub(".csv", "", lang))

# The generated strings were padded with numeric suffixes to reach the requested
# number of entries (e.g. "(no place)_286"). Remove these and deduplicate.
# Religious life (e.g. "friar", "lay sister") is dropped: HISCO has codes for
# these (e.g. 14140), so they are not 'no occupation'. Category names are
# translated in the Norwegian file.
no_occupation = no_occupation %>%
    filter(!is.na(occ1)) %>%
    filter(!category %in% c("religious_life", "religiost_liv")) %>%
    mutate(occ1 = str_remove(occ1, "_[0-9]+$") %>% tolower()) %>% # Lower case before deduplication (done below anyway)
    distinct(lang, occ1)


# ==== Farmer & Fisherman ====
set.seed(20)
# Plane version:
fisherman_and_farmers = fisherman_and_farmers %>%
    select(Code, Farmer_and_Fisherman, Fisherman_and_Farmer) %>%
    mutate(hisco_1 = "61110", hisco_2 = "64100") %>%
    pivot_longer(cols = c("Farmer_and_Fisherman", "Fisherman_and_Farmer"), names_to = "var", values_to = "occ1") %>%
    select(-var) %>%
    rename(lang = Code) %>%
    sample_n(2000, replace = TRUE) # Upsample to 2000 obs

# Variations
fisherman_and_farmer_variations = fisherman_and_farmer_variations %>%
    mutate(hisco_1 = "61110", hisco_2 = "64100")

# ==== No occupation ====
no_occupation = no_occupation %>%
    mutate(hisco_1 = "-1", hisco_2 = " ")

data0 = bind_rows(fisherman_and_farmers, fisherman_and_farmer_variations, no_occupation)

# ==== More cleaning ====
data0 = data0 %>%
  filter(!is.na(occ1), str_trim(occ1) != "") %>%
  mutate( # Clean string:
    occ1 = occ1 %>% tolower()
  ) %>%
  mutate(
    hisco_1 = as.numeric(hisco_1),
    hisco_2 = as.numeric(hisco_2)
  ) %>% 
  mutate(
    hisco_1 = ifelse(is.na(hisco_1), -1, hisco_1)
  ) %>% 
  mutate(
    hisco_2 = ifelse(is.na(hisco_2), " ", hisco_2)
  )

data1 = data0 %>%
  mutate(
    hisco_3 = " ",
    hisco_4 = " ",
    hisco_5 = " "
  )

# ==== Check against authoritative HISCO list ====
load("Data/Key.Rdata")

key = key %>% select(hisco, code)

# Remove data not in key (erronoeous data somehow)
data1 %>% 
  filter(!hisco_1 %in% key$hisco)

data1 %>% 
  filter(!hisco_2 %in% key$hisco)

n1 = NROW(data1)
data1 = data1 %>% 
  filter(hisco_1 %in% key$hisco) %>% 
  filter(hisco_2 %in% key$hisco) %>% 
  filter(hisco_3 %in% key$hisco) %>% 
  filter(hisco_4 %in% key$hisco) %>% 
  filter(hisco_5 %in% key$hisco)

NROW(data1) - n1 # 0 observations

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
  mutate(RowID = paste0("Handheld", 1:n()))

# Extra columns (found in all training data)
data1 = data1 %>%
    mutate(
        split = "Train",
        Year = NA
    )

set.seed(20)
data1 %>% sample_frac(1)

# ==== Save ====
data1 %>% write_csv("Data/Training_data/Handheld_data_train.csv")


