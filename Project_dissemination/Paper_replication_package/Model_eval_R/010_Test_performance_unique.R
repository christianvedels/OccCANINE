# Test model performance on unique test data
# Created:  2025-11-11
# Authors:  Christian Vedel [christian-vs@sam.sdu.dk],
#
# Purpose:  Make pretty tables of test data performance for the paper

# ==== Libraries ====
library(tidyverse)
library(foreach)
library(knitr)
library(fixest)

# ==== Load data ====
files = list.files("Project_dissemination/Paper_replication_package/Data/Intermediate_data/test_unique_performance", full.names = TRUE)
files = files[!grepl("Data/Intermediate_data/test_unique_performance/lang", files)]  # Exclude language folder
files = files[!grepl("Data/Intermediate_data/test_unique_performance/source", files)]  # Exclude source folder
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
sink("Project_dissemination/Paper_replication_package/Tables/Test_performance_unique.txt", append = FALSE)
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


# # ==== Individual tables ==== # Cover later
# unique_obs = read_csv("Project_dissemination/Paper_replication_package/Data/Intermediate_data/big_files/obs_test_unique_performance_greedy.csv")
# obs = read_csv("Project_dissemination/Paper_replication_package/Data/Intermediate_data/big_files/obs_test_performance_greedy.csv")


# obs = obs %>%
#     mutate(
#         occ_string = # "ca[SEP]sargento..." --> "sargento ..."
#         str_replace(occ1, "^.*?\\[SEP\\]", "")
#     ) %>%
#     mutate(
#         string_length = nchar(as.character(occ_string))
#     )
    
# obs %>%
#     group_by(string_length) %>%
#     summarise(
#         n = n(),
#         accuracy = mean(acc),
#     ) %>%
#     filter(n>30) %>%
#     ggplot(aes(x = string_length, y = accuracy)) +
#     geom_point() +
#     geom_smooth(method = "loess") +
#     labs(
#         title = "Model accuracy by string length on full test set",
#         x = "String length (number of characters)",
#         y = "Accuracy"
#     ) +
#     theme_minimal()

# unique_obs = unique_obs %>%
#     mutate(
#         occ_string = # "ca[SEP]sargento..." --> "sargento ..."
#         str_replace(occ1, "^.*?\\[SEP\\]", "")
#     ) %>%
#     mutate(
#         string_length = nchar(as.character(occ_string))
#     )
    
    
# unique_obs %>%
#     group_by(string_length) %>%
#     summarise(
#         n = n(),
#         accuracy = mean(acc),
#     ) %>%
#     filter(n>30) %>%
#     ggplot(aes(x = string_length, y = accuracy)) +
#     geom_point() +
#     geom_smooth(method = "loess") +
#     labs(
#         title = "Model accuracy by string length on full test set",
#         x = "String length (number of characters)",
#         y = "Accuracy"
#     ) +
#     theme_minimal()


# complete_data = unique_obs %>%
#     mutate(
#         type = "unique"
#     ) %>%
#     bind_rows(
#         obs %>% mutate(
#             type = "all"
#         )
#     )

# demean0 = function(x) {
#     x - mean(x, na.rm = TRUE)
# }

# p1 = complete_data %>%
#     filter(string_length <= 128) %>%
#     group_by(lang) %>%
#     mutate(
#         acc = demean0(acc)
#     ) %>%
#     ggplot(aes(x = string_length, y = acc, col = type)) +
#     geom_smooth() +
#     labs(
#         title = "Model accuracy by string length on unique vs all test data",
#         x = "String length (number of characters)",
#         y = "Accuracy",
#         col = "Data type"
#     ) +
#     theme_bw()

# ggsave("Project_dissemination/Paper_replication_package/Figures/Accuracy_string_length_unique_vs_all.png", plot = p1, width = 8, height = 6)

# # Residualized by string length
# smooth.spline(obs$string_length, obs$acc) -> ss_acc
# unique_obs = unique_obs %>%
#     mutate(
#         acc_resid = acc - predict(ss_acc, string_length)$y
#     )

# p2 = unique_obs %>% ggplot(aes(x = string_length, y = acc_resid)) +
#     geom_smooth() +
#     labs(
#         title = "Residualized model accuracy by string length on unique test data",
#         x = "String length (number of characters)",
#         y = "Residualized Accuracy"
#     ) +
#     theme_bw()

# ggsave("Project_dissemination/Paper_replication_package/Figures/Accuracy_string_length_unique_residualized.png", plot = p2, width = 8, height = 6)


# feols(acc ~ type + string_length*type + I(string_length^2)*type + I(string_length^3)*type | lang, data = complete_data) %>% summary()
# feols(precision ~ type + string_length*type + I(string_length^2)*type + I(string_length^3)*type | lang, data = complete_data) %>% summary()
# feols(recall ~ type + string_length*type + I(string_length^2)*type + I(string_length^3)*type | lang, data = complete_data) %>% summary()
# feols(f1 ~ type + string_length*type + I(string_length^2)*type + I(string_length^3)*type | lang, data = complete_data) %>% summary()


