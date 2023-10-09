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
data0 = read_csv2(
  "Data/Raw_data/SWEDPOP_data/individual.csv", 
  locale = locale(encoding = "ISO-8859-1")
)

# ==== Cleaning data ====
# data0$TYPE %>% unique()
# [1] "ARRIVAL_FROM"                           "PERSON_IN_LABOUR_FORCE"                
# [3] "PERSON_IS_ABSENT"                       "BIRTH_DATE"                            
# [5] "BIRTH_LOCATION"                         "CIVIL_STATUS"                          
# [7] "OCCUPATION_HISCO_PART_1"                "OCCUPATION_HISCO_PRODUCT_PART_1"       
# [9] "OCCUPATION_HISCO_RELATION_PART_1"       "OCCUPATION_HISCO_STATUS_PART_1"        
# [11] "BIRTH_RESIDENCE"                        "OCCUPATION_HISCO_PART_2"               
# [13] "OCCUPATION_HISCO_PRODUCT_PART_2"        "OCCUPATION_HISCO_RELATION_PART_2"      
# [15] "OCCUPATION_HISCO_STATUS_PART_2"         "REASON_FOR_BEING_ABSENT"               
# [17] "DISABILITY_AS_TRANSCRIBED"              "DISABILITY_CODE_1"                     
# [19] "NOT_BAPTISED"                           "TITLE_AS_TRANSCRIBED"                  
# [21] "OCCUPATION_HISCO_PART_3"                "OCCUPATION_HISCO_PRODUCT_PART_3"       
# [23] "OCCUPATION_HISCO_RELATION_PART_3"       "OCCUPATION_HISCO_STATUS_PART_3"        
# [25] "DISABILITY_CODE_2"                      "BAPTISM_DATE"                          
# [27] "OCCUPATION_HISCO_PART_4"                "OCCUPATION_HISCO_PRODUCT_PART_4"       
# [29] "OCCUPATION_HISCO_RELATION_PART_4"       "OCCUPATION_HISCO_STATUS_PART_4"        
# [31] "BAPTISM_LOCATION"                       "STILLBIRTH_DATE"                       
# [33] "DISABILITY_CODE_3"                      "STILLBIRTH_LOCATION"                   
# [35] "CHILDBIRTH_ASSISTANT"                   "HAS_LEFT_THE_SWEDISH_CHURCH"           
# [37] "OCCUPATION_HISCO_PART_5"                "OCCUPATION_HISCO_PRODUCT_PART_5"       
# [39] "OCCUPATION_HISCO_RELATION_PART_5"       "OCCUPATION_HISCO_STATUS_PART_5"        
# [41] "DEATH_DATE"                             "OCCUPATION_STANDARD_PART_1"            
# [43] "START_OBSERVATION"                      "DEPARTURE_TO"                          
# [45] "BIRTH_OCCUPATION_HISCO_PART_1"          "BIRTH_OCCUPATION_HISCO_STATUS_PART_1"  
# [47] "BIRTH_OCCUPATION_HISCO_RELATION_PART_1" "MARRIAGE_DATE"                         
# [49] "END_OBSERVATION"                        "INCOME_TAXED"                          
# [51] "BIRTH_OCCUPATION_HISCO_STATUS_PART_2"   "INCOME_AGRICULTURAL_PROPERTY"          
# [53] "INCOME_CAPITAL"                         "INCOME_IMMOVABLE_PROPERTY"             
# [55] "INCOME_LABOUR"                          "INCOME_SELF_EMPLOYMENT"                
# [57] "INCOME_TOTAL"                           "DEATH_CAUSE_ICD10H_PART_1"             
# [59] "INCOME_OTHER_PROPERTY"                  "DEATH_LOCATION"                        
# [61] "OCCUPATION_STANDARD_PART_2"             "BIRTH_OCCUPATION_HISCO_PART_2"         
# [63] "BIRTH_OCCUPATION_HISCO_RELATION_PART_2" "DEATH_CAUSE_ICD10H_PART_2"             
# [65] "OCCUPATION_STANDARD_PART_3"             "DEATH_CAUSE_ICD10H_PART_3"             
# [67] "BIRTH_OCCUPATION_HISCO_STATUS_PART_3"   "DEATH_CAUSE_ICD10H_PART_4"             
# [69] "BIRTH_OCCUPATION_HISCO_PART_3"          "BIRTH_OCCUPATION_HISCO_RELATION_PART_3"
# [71] "DEATH_CAUSE_ICD10H_PART_5"              "DIVORCE_DATE"                          
# [73] "FUNERAL_DATE"                           "FUNERAL_LOCATION"                      
# [75] "LEGITIMACY"                             "MARRIAGE_LOCATION"                     
# [77] "MARRIAGE_SEQUENCE"                      "MULTIPLE_BIRTH"                        
# [79] "OCCUPATION_STANDARD_PART_4"             "OCCUPATION_STANDARD_PART_5"            
# [81] "OCCUPATION_HISCO_RELATION_PART_6"       "OCCUPATION_HISCO_STATUS_PART_6"        
# [83] "OCCUPATION_HISCO_PART_6"                "OCCUPATION_STANDARD_PART_6"            
# [85] "OCCUPATION_HISCO_PRODUCT_PART_6"        "VACCINATION"                           
# [87] "DEATH_RESIDENCE" 
data0 = data0 %>% 
  filter(
    TYPE %in% c(
      "BIRTH_OCCUPATION_HISCO_PART_1",
      "BIRTH_OCCUPATION_HISCO_PART_2",
      "BIRTH_OCCUPATION_HISCO_PART_3",
      
      # # Status
      # "BIRTH_OCCUPATION_HISCO_STATUS_PART_1",
      # "BIRTH_OCCUPATION_HISCO_STATUS_PART_2",
      # "BIRTH_OCCUPATION_HISCO_STATUS_PART_3",
      # 
      # # Relation
      # "BIRTH_OCCUPATION_HISCO_RELATION_PART_1",
      # "BIRTH_OCCUPATION_HISCO_RELATION_PART_2",
      # "BIRTH_OCCUPATION_HISCO_RELATION_PART_3",
      
      # HISCO
      "OCCUPATION_HISCO_PART_1",
      "OCCUPATION_HISCO_PART_2",
      "OCCUPATION_HISCO_PART_3",
      "OCCUPATION_HISCO_PART_4",
      "OCCUPATION_HISCO_PART_5",
      "OCCUPATION_HISCO_PART_6",
      
      # # Status
      # "OCCUPATION_HISCO_STATUS_PART_1",
      # "OCCUPATION_HISCO_STATUS_PART_2",
      # "OCCUPATION_HISCO_STATUS_PART_3",
      # "OCCUPATION_HISCO_STATUS_PART_4",
      # "OCCUPATION_HISCO_STATUS_PART_5",
      # "OCCUPATION_HISCO_STATUS_PART_6",
      # 
      # # Product 
      # "OCCUPATION_HISCO_PRODUCT_PART_1",
      # "OCCUPATION_HISCO_PRODUCT_PART_2",
      # "OCCUPATION_HISCO_PRODUCT_PART_3",
      # "OCCUPATION_HISCO_PRODUCT_PART_4",
      # "OCCUPATION_HISCO_PRODUCT_PART_5",
      # "OCCUPATION_HISCO_PRODUCT_PART_6",
      # 
      # # Relation
      # "OCCUPATION_HISCO_RELATION_PART_1",
      # "OCCUPATION_HISCO_RELATION_PART_2",
      # "OCCUPATION_HISCO_RELATION_PART_3",
      # "OCCUPATION_HISCO_RELATION_PART_4",
      # "OCCUPATION_HISCO_RELATION_PART_5",
      # "OCCUPATION_HISCO_RELATION_PART_6",
      
      # Cleaned title
      "OCCUPATION_STANDARD_PART_1",
      "OCCUPATION_STANDARD_PART_2",
      "OCCUPATION_STANDARD_PART_3",
      "OCCUPATION_STANDARD_PART_4",
      "OCCUPATION_STANDARD_PART_5",
      "OCCUPATION_STANDARD_PART_6",
    
      # Raw
      "TITLE_AS_TRANSCRIBED",
      "CIVIL_STATUS"
    )
  ) %>% 
  select(SOURCE, ID_I, TYPE, YEAR, VALUE) %>% 
  pivot_wider(names_from = TYPE, values_from = VALUE)

# data0 %>% 
#   drop_na(OCCUPATION_STANDARD_PART_1) %>% View()

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
