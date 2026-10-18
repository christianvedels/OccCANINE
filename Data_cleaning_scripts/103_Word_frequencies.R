# Word frequencies
# Created:    2026-10-18
# Auhtors:    Christian Vedel [christian-vs@sam.sdu.dk],
#
# Purpose:    Counts how often each word occurs in the reweighted training
#             data. Used in training (train_mixer.py --fn-word-freq) to draw
#             realistic random words for the 'insert random word' augmentation.
#             Words are counted per row, so frequent strings weigh more.
#             augmentations.csv is not included (typos and gibberish).
#
# Output:     Data/word_freqs.csv with columns "word" and "freq"

# ==== Libraries ====
library(tidyverse)
library(foreach)

# ==== Count words by file ====
files = list.files("Data/Training_data_v2", full.names = TRUE, pattern = "_train\\.csv$")

word_freqs = foreach(f = files, .combine = "bind_rows") %do% {
  cat("\nCounting words in", f)
  read_csv(f, col_select = occ1, col_types = cols(.default = "c"), progress = FALSE, trim_ws = FALSE) %>%
    filter(!is.na(occ1)) %>%
    mutate(word = str_split(occ1, "\\s+")) %>%
    select(word) %>%
    unnest(word) %>%
    filter(word != "") %>%
    count(word, name = "freq")
}

word_freqs = word_freqs %>%
  group_by(word) %>%
  summarise(freq = sum(freq), .groups = "drop") %>%
  arrange(desc(freq))

# ==== Save ====
word_freqs %>% write_csv("Data/word_freqs.csv")

cat("\n\nFiles:", length(files))
cat("\nUnique words:", NROW(word_freqs))
cat("\nTotal words:", sum(word_freqs$freq), "\n")
