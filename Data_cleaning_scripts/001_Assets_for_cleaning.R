# Assets for all data cleaning
# Created:    2023-05-23
# Authors:    Christian Vedel [christian-vs@sam.sdu.dk],
#
# Purpose:    Produces common keys, etc to be used across all data
#
# Output:     Key.csv and Key.Rdata

# ==== Libraries ====
# library(devtools)
# install_github("junkka/hisco")
library(tidyverse)

# ==== Load data ====

# ==== Produce key ====
key = hisco::hisco %>% 
  distinct(en_hisco_text, hisco)

# Check uniqueness
key %>% count(hisco) %>% filter(n>1)
key %>% count(en_hisco_text) %>% filter(n>1)
key %>% filter(en_hisco_text %in% c("Buyers", "Other Painters, Construction"))
# Conclusion: Not a problem

# Add codes to key
key = key %>% 
  mutate(code = seq(0, n()-1, by = 1))

# Add code for empty
key = bind_rows(
  data.frame(
    en_hisco_text = "empty",
    hisco = " ",
    code = NA
  ),
  key %>% mutate(hisco = as.character(hisco))
)

key %>% 
  write_csv("Data/Key.csv")

save(key, file = "Data/Key.Rdata")
