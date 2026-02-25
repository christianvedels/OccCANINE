# Descriptive statistics
# Created:  2025-03-17
# Authors:  Christian Vedel [christian-vs@sam.sdu.dk],
#
# Purpose:  Descriptive statistics used in the paper

# ==== Libraries ====
library(tidyverse)
library(foreach)
library(knitr)
library(kableExtra)
source("Project_dissemination/Paper_replication_package/Model_eval_R/000_Functions.R")

# ==== Load data ====
train = read0("Data/Training_data")
val1 = read0("Data/Validation_data1")
val2 = read0("Data/Validation_data2")
test = read0("Data/Test_data")

# ==== Amount of data ====
data_summary = data.frame(
    Dataset = c("Training", "Validation1", "Validation2", "Test"),
    Obs = c(NROW(train), NROW(val1), NROW(val2), NROW(test))
)
print(data_summary)

sink("Project_dissemination/Paper_replication_package/Tables/Data_split_summary.txt")
data_summary %>% mutate(Pct = Obs / sum(Obs)) %>% print()
sink()

# Clean f name
clean_f_name = function(x){
    x = gsub(".csv", "", x)
    x = gsub("_train", "", x)
    x = gsub("_val1", "", x)
    x = gsub("_val2", "", x)
    x = gsub("_test", "", x)
    return(x)
}

# === Table: Data by data source ====
tmp1 = train %>%
    group_by(file) %>%
    summarise(
        train = n(),
        Language = ifelse(length(unique(lang))==1, unique(lang), "mix")
    ) %>%
    ungroup() %>%
    mutate(
        file = clean_f_name(file)
    )

tmp2 = val1 %>%
    group_by(file) %>%
    summarise(
        val1 = n()
    ) %>%
    ungroup() %>%
    mutate(
        file = clean_f_name(file)
    )

tmp3 = val2 %>%
    group_by(file) %>%
    summarise(
        val2 = n()
    ) %>%
    ungroup() %>%
    mutate(
        file = clean_f_name(file)
    )

tmp4 = test %>%
    group_by(file) %>%
    summarise(
        test = n()
    ) %>%
    ungroup() %>%
    mutate(
        file = clean_f_name(file)
    )


sink("Project_dissemination/Paper_replication_package/Tables/Data_summary.txt")
tmp1 %>% 
    full_join(tmp2, by = "file") %>%
    full_join(tmp3, by = "file") %>%
    full_join(tmp4, by = "file") %>%
    mutate(
        Observations = train + val1 + val2 + test
    ) %>%
    select(file, Observations, Language) %>%
    mutate(
        Percent = Observations / sum(Observations)
    ) %>%
    arrange(desc(Observations)) %>%
    mutate(
        Observations = scales::comma(Observations),
        Percent = scales::percent(Percent, accuracy = 0.01)
    ) %>%
    mutate(
        Source = " "
    ) %>%
    rename(
        `Shorthand name` = file
    ) %>%
    kable(
        format = "latex",
        booktabs = TRUE,
        caption = "Data summary",
        linesep = ""
    ) %>% print()
sink()

# === Table: Data by language ====
tmp1 = train %>%
    group_by(lang) %>%
    summarise(
        train = n()
    ) %>%
    ungroup()

tmp2 = val1 %>%
    group_by(lang) %>%
    summarise(
        val1 = n()
    ) %>%
    ungroup()

tmp3 = val2 %>%
    group_by(lang) %>%
    summarise(
        val2 = n()
    ) %>%
    ungroup()

tmp4 = test %>%
    group_by(lang) %>%
    summarise(
        test = n()
    ) %>%
    ungroup()

sink("Project_dissemination/Paper_replication_package/Tables/Data_summary_lang.txt")
tmp1 %>% 
    full_join(tmp2, by = "lang") %>%
    full_join(tmp3, by = "lang") %>%
    full_join(tmp4, by = "lang") %>%
    mutate(
        Observations = train + val1 + val2 + test
    ) %>%
    rename(
        Language = lang
    ) %>%
    select(Language, Observations) %>%
    mutate(
        Percent = Observations / sum(Observations)
    ) %>%
    arrange(desc(Observations)) %>%
    mutate(
        Observations = scales::comma(Observations),
        Percent = scales::percent(Percent, accuracy = 0.01)
    ) %>%
    mutate(
        Source = " "
    ) %>%
    kable(
        format = "latex",
        booktabs = TRUE,
        caption = "Data summary",
        linesep = ""
    ) %>% print()
sink()

# === Table: String length descriptive statistics ====
get_string_length_stats = function(df, dataset_name) {
    string_length = nchar(as.character(df$occ1))
    string_length = string_length[!is.na(string_length)]
    qs = quantile(string_length, probs = c(0.25, 0.5, 0.75, 0.99, 0.999), na.rm = TRUE, names = FALSE)
    pct_120 = mean(string_length <= 120, na.rm = TRUE)
    data.frame(
        Dataset = dataset_name,
        P25 = qs[1],
        Median = qs[2],
        P75 = qs[3],
        P99 = qs[4],
        P99_9 = qs[5],
        `Pct.` = pct_120
    )
}

string_length_summary = bind_rows(
    get_string_length_stats(train, "Training"),
    get_string_length_stats(val1, "Validation1"),
    get_string_length_stats(val2, "Validation2"),
    get_string_length_stats(test, "Test"),
    get_string_length_stats(bind_rows(train, val1, val2, test), "All")
) %>%
    mutate(
        across(c(P25, Median, P75, P99, P99_9), ~ round(., 1)),
        `Pct.` = scales::percent(`Pct.`, accuracy = 0.01)
    )

sink("Project_dissemination/Paper_replication_package/Tables/String_length_summary.txt")
string_length_summary %>%
    kable(
        format = "latex",
        booktabs = TRUE,
        caption = "String length descriptive statistics",
        linesep = ""
    ) %>% print()
sink()
