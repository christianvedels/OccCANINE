# Test model performance
# Created:  2025-04-10
# Authors:  Christian Vedel [christian-vs@sam.sdu.dk],
#
# Purpose:  Make pretty tables of test data performance for the paper

# ==== Libraries ====
library(tidyverse)
library(foreach)
library(knitr)

# ==== Load data ====
files = list.files("Project_dissemination/Paper_replication_package/Data/Intermediate_data/test_performance", full.names = TRUE)
files = files[!grepl("Data/Intermediate_data/test_performance/lang", files)]  # Exclude language folder
files = files[!grepl("Data/Intermediate_data/test_performance/source", files)]  # Exclude source folder
foreach(f = files, .combine = "bind_rows") %do% {
    read_csv(f, show_col_types = FALSE) %>%
        mutate(file = f) %>%
        mutate(file = gsub("Project_dissemination/Paper_replication_package/Data/Intermediate_data/test_performance/", "", file)) %>%
        mutate(
            digits = case_when(
                grepl("digits_1", file) ~ 1,
                grepl("digits_2", file) ~ 2,
                grepl("digits_3", file) ~ 3,
                grepl("digits_4", file) ~ 4,
                grepl("digits_5", file) ~ 5
            )
        )
} -> test_performance


# ==== Data preparation ====
performance_table = test_performance %>% pivot_longer(
    cols = c("accuracy", "precision", "recall", "f1"), 
    names_to = "statistic", 
    values_to = "value"
) %>% 
    select(-file) %>%
    pivot_wider(
        names_from = "digits", 
        values_from = "value"
    ) %>%
    select(
        prediction_type, statistic, lang,
        `1`, `2`, `3`, `4`, `5`,
        n
    ) %>%
    arrange(
        case_when(
            prediction_type == "greedy" ~ 1,
            prediction_type == "flat" ~ 2,
            prediction_type == "full" ~ 3,
        )
    ) %>%
    mutate(
        lang = ifelse(lang == "known", "With lang. info.", "Without lang. info."),
    ) %>%
    mutate(
        # Rename to 'fast' 'good' and 'full'
        prediction_type = case_when(
            prediction_type == "greedy" ~ "good",
            prediction_type == "flat" ~ "fast",
            prediction_type == "full" ~ "full"
        ),
    )
# Make LaTeX table with knitr::kable
sink("Project_dissemination/Paper_replication_package/Tables/Test_performance.txt", append = FALSE)
cat("\nTest set performance by prediction type and number of digits")
cat("\nTest performance with language information\n")
performance_table %>%
    filter(lang == "With lang. info.") %>%
    filter(prediction_type %in% c("good", "fast")) %>%
    select(-lang) %>%
    kable(
        format = "latex",
        booktabs = TRUE,
        digits = 3,
        caption = "Test set performance by prediction type and number of digits"
    ) %>% print()

cat("\n")
cat("Test performance without language information:\n")
performance_table %>%
    filter(lang == "Without lang. info.") %>%
    filter(prediction_type %in% c("good", "fast")) %>%
    select(-lang) %>%
    kable(
        format = "latex",
        booktabs = TRUE,
        digits = 3,
        caption = "Test set performance by prediction type and number of digits (without language information)"
    ) %>% print()
sink()

# Lang info performance boost
sink("Project_dissemination/Paper_replication_package/Tables/Test_performance_lang_info_boost.txt", append = FALSE)
cat("\nTest set performance boost with language information\n")
test_performance %>%
    filter(digits == 5) %>%
    filter(prediction_type %in% c("flat", "greedy")) %>%
    arrange(
        case_when(
            prediction_type == "greedy" ~ 1,
            prediction_type == "flat" ~ 2,
            prediction_type == "full" ~ 3
        )
    ) %>%
    select(prediction_type, lang, accuracy, precision, recall, f1, n) %>%
    pivot_longer(
        cols = c(accuracy, precision, recall, f1),
        names_to = "statistic",
        values_to = "value"
    ) %>%
    pivot_wider(
        names_from = lang,
        values_from = value
    ) %>%
    mutate(dif = known - unk) %>%
    pivot_longer(
        c(known, unk, dif),
        names_to = c("lang"),
    ) %>%
    pivot_wider(
        names_from = statistic,
        values_from = value
    ) %>%
    select(
        prediction_type,
        lang, 
        accuracy, precision, recall, f1, n
    ) %>%
    kable(
        format = "latex",
        booktabs = TRUE,
        digits = 3
    ) %>% print()
sink()

# Save optimal performance
test_performance %>%
    filter(digits == 5) %>%
    filter(prediction_type %in% c("greedy")) %>%
    filter(lang == "known") %>%
    write_csv(
        "Project_dissemination/Paper_replication_package/Data/Intermediate_data/test_performance_greedy_wlang.csv"
    )
