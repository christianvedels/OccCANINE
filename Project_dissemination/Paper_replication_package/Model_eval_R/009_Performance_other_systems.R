# Descriptive stats for other systems
# Created:  2025-09-01
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
tl = FALSE  # Toyload for quick testing
files_train = c("EN_ISCO68_IPUMS_UK_train.csv", "EN_ISCO68_IPUMS_UK_n_unq10000_train.csv", "EN_OCCICEM_IPUMS_UK_train.csv", "EN_OCCICEM_IPUMS_UK_n_unq10000_train.csv", "EN_OCC1950_IPUMS_US_train.csv", "EN_OCC1950_IPUMS_US_n_unq10000_train.csv")
files_val1 = c("EN_ISCO68_IPUMS_UK_val1.csv", "EN_ISCO68_IPUMS_UK_n_unq10000_val1.csv", "EN_OCCICEM_IPUMS_UK_val1.csv", "EN_OCCICEM_IPUMS_UK_n_unq10000_val1.csv", "EN_OCC1950_IPUMS_US_val1.csv", "EN_OCC1950_IPUMS_US_n_unq10000_val1.csv")
files_val2 = c("EN_ISCO68_IPUMS_UK_val2.csv", "EN_ISCO68_IPUMS_UK_n_unq10000_val2.csv", "EN_OCCICEM_IPUMS_UK_val2.csv", "EN_OCCICEM_IPUMS_UK_n_unq10000_val2.csv", "EN_OCC1950_IPUMS_US_val2.csv", "EN_OCC1950_IPUMS_US_n_unq10000_val2.csv")
files_test = c("EN_ISCO68_IPUMS_UK_test.csv", "EN_OCCICEM_IPUMS_UK_test.csv", "EN_OCC1950_IPUMS_US_test.csv")

train = read0("Data/Training_data_other", verbose = TRUE, files = files_train, toyload = tl)
val1 = read0("Data/Validation_data1_other", verbose = TRUE, files = files_val1, toyload = tl)
val2 = read0("Data/Validation_data2_other", verbose = TRUE, files = files_val2, toyload = tl)
test = read0("Data/Test_data_other", verbose = TRUE, files = files_test, toyload = tl)

# ==== Amount of data ====
data_summary = data.frame(
    Dataset = c("Training", "Validation1", "Validation2", "Test"),
    Obs = c(NROW(train), NROW(val1), NROW(val2), NROW(test))
)
print(data_summary)

sink("Project_dissemination/Paper_replication_package/Tables/Data_split_summary_other_systems.txt")
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
        train_unique = length(unique(occ1)),
        Language = ifelse(length(unique(lang))==1, unique(lang), "mix")
    ) %>%
    ungroup() %>%
    mutate(
        file = clean_f_name(file)
    )

tmp2 = val1 %>%
    group_by(file) %>%
    summarise(
        val1 = n(),
        val1_unique = length(unique(occ1))
    ) %>%
    ungroup() %>%
    mutate(
        file = clean_f_name(file)
    )

tmp3 = val2 %>%
    group_by(file) %>%
    summarise(
        val2 = n(),
        val2_unique = length(unique(occ1))
    ) %>%
    ungroup() %>%
    mutate(
        file = clean_f_name(file)
    )

tmp4 = test %>%
    group_by(file) %>%
    summarise(
        test = n(),
        test_unique = length(unique(occ1))
    ) %>%
    ungroup() %>%
    mutate(
        file = clean_f_name(file)
    )


sink("Project_dissemination/Paper_replication_package/Tables/Data_summary_other.txt")
tmp1 %>% 
    full_join(tmp2, by = "file") %>%
    full_join(tmp3, by = "file") %>%
    full_join(tmp4, by = "file") %>%
    mutate(
        Observations = train + val1 + val2 + test,
        `Unique strings (training)` = train_unique # Only training unique strings reported
    ) %>%
    select(file, Observations, `Unique strings (training)`, Language) %>%
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

