# Descriptive statistics
# Created:  2025-03-20
# Authors:  Christian Vedel [christian-vs@sam.sdu.dk],
#
# Purpose:  Eval stats from ood data

# ==== Libraries ====
library(tidyverse)
library(foreach)

# ==== Load data ====
files = list.files("Data/OOD_data/Predictions", full.names = TRUE)

ood_performance = foreach(f = files, .combine = "bind_rows") %do% {
    if(f %in% c(
        "Data/OOD_data/Predictions/DA_Copenhagen_Burial_Records_manually_checked.xlsx",
        "Data/OOD_data/Predictions/predictions_DA_Copenhagen_Burial_Records.csv"
    )) {
        return(NULL)  # Skip these files
    }

    read_csv(f, show_col_types = FALSE) %>%
        mutate(file = f) %>%
        mutate(file = gsub("Data/OOD_data/Predictions/", "", file)) %>%
        select(-`...1`, -occ1, -conf, -acc, -precision, -recall, -f1) %>%
        pivot_longer(
            cols = acc_1:f1_5,
            names_to = c("statistic", "digits"),
            values_to = "value",
            names_pattern = "(.*)_(\\d+)"
        ) %>%
        group_by(file, statistic, digits) %>%
        summarise(
            value = mean(value, na.rm = TRUE),
            file = file[1],
            n = n()
        )
}

ood_performance %>% 
    pivot_wider(
        names_from = digits,
        values_from = value
    )
