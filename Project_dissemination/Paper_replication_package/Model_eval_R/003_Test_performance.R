# Test model performance
# Created:  2025-04-10
# Authors:  Christian Vedel [christian-vs@sam.sdu.dk],
#
# Purpose:  Test model performance

# ==== Libraries ====
library(tidyverse)
library(foreach)

# ==== Load data ====
fs = list.files("Data/eval-results/hisco", full.names = TRUE)
all_data = foreach(f = fs, .combine = "bind_rows") %do% {
    cat(paste0("Reading ", f, "\n"))
    read_csv(f, guess_max = 100000) %>% 
        mutate(file = f) %>%
        mutate_all(as.character) %>%
        sample_n(100, replace=TRUE) # Sample for development
}

all_data %>%

    mutate(
        unspec_lang = grepl("l=None", file),
        prediction_type = case_when(
            grepl("pt=greedy", file) ~ "greedy",
            grepl("pt=flat", file) ~ "flat",
            grepl("pt=full", file) ~ "full"
        ),
    ) %>% 
    group_by(prediction_type, unspec_lang, lang) %>%
    summarise(
        acc = mean(as.numeric(acc)),
        precision = mean(as.numeric(precision)),
        recall = mean(as.numeric(recall)),
        f1 = mean(as.numeric(f1)),
        n = n()
    ) %>%
    ggplot(aes(lang, f1, fill = prediction_type)) +
    geom_bar(stat = "identity", position = "dodge") +
    facet_wrap(~unspec_lang)
