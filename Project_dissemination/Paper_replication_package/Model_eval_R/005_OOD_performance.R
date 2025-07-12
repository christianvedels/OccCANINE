# Descriptive statistics
# Created:  2025-03-20
# Authors:  Christian Vedel [christian-vs@sam.sdu.dk],
#
# Purpose:  Eval stats from ood data

# ==== Libraries ====
library(tidyverse)
library(foreach)
library(knitr)

# ==== Load data ====
files = list.files("Data/OOD_data/Predictions", full.names = TRUE)

ood_performance = foreach(f = files, .combine = "bind_rows") %do% {
    if(f %in% c(
        "Data/OOD_data/Predictions/Manually_checked"
    )) {
        return(NULL)  # Skip these files
    }

    res = read_csv(f, show_col_types = FALSE) %>%
        mutate(file = f)

    if(all(is.na(res$acc))) {
        return(NULL)  # Skip files with no accuracy data
    }
        
    res %>%
        mutate(file = gsub("Data/OOD_data/Predictions/", "", file)) %>%
        select(-`...1`, -occ1, -acc, -precision, -recall, -f1) %>%
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

res1 = ood_performance %>% 
    pivot_wider(
        names_from = digits,
        values_from = value
    ) %>%
    mutate(tmp = -`1`) %>%
    arrange(statistic, tmp, file) %>%
    select(-tmp)

# Manually checked files
files = list.files("Data/OOD_data/Predictions/Manually_checked", full.names = TRUE, pattern = "\\.csv$")

ood_performance_manual = foreach(f = files, .combine = "bind_rows") %do% {
    res = read_csv2(f, show_col_types = FALSE) %>%
        mutate(file = f)
        
    res %>%
        mutate(file = gsub("Data/OOD_data/Predictions/Manually_checked/", "", file)) %>%
        select(Check_substantial, Check_strict, file, n) %>%
        pivot_longer(
            cols = Check_substantial:Check_strict,
            names_to = c("statistic"),
            values_to = "value"
        ) %>%
        group_by(file, statistic) %>%
        summarise(
            value = weighted.mean(value, n, na.rm = TRUE),
            file = file[1],
            n = n()
        )
}

res2 = ood_performance_manual %>%
    mutate(statistic = recode(statistic, Check_substantial = "substantial", Check_strict = "strict")) %>%
    arrange(statistic, file) %>%
    pivot_wider(
        names_from = statistic,
        values_from = value
    ) %>%
    mutate(file = gsub("Data/OOD_data/Predictions/Manually_checked/", "", file))


# Latex in two panels printed to Tables/ood_performance.tex
# Panel A: Manually checked
sink("Project_dissemination/Paper_replication_package/Tables/ood_performance.txt")
res2 %>%
    mutate_if(is.numeric, ~ round(., 3)) %>%
    select(file, n, substantial, strict) %>%
    kable(
        format = "latex",
        booktabs = TRUE,
        caption = "Manually checked OOD data performance"
    ) %>% print()

# Panel B: Data which is already HISCO coded
res1 %>%
    mutate_if(is.numeric, ~ round(., 3)) %>%
    filter(statistic == "acc") %>%
    kable(
        format = "latex",
        booktabs = TRUE,
        caption = "OOD data performance",
        col.names = c("File", "Statistic", "n", "1 digit", "2 digits", "3 digits", "4 digits", "5 digits")
    ) %>% print()

sink()



