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
read0 = function(dir){
  fs = list.files(dir)
  fs = fs[grepl(".csv", fs)]
  foreach(f = fs, .combine = "bind_rows") %do% {
    read_csv(paste0(dir,"/",f), guess_max = 100000) %>% mutate(file = f)
  }
}

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

data_summary %>% mutate(Obs / sum(Obs))

# Clean f name
clean_f_name = function(x){
    x = gsub(".csv", "", x)
    x = gsub("_train", "", x)
    x = gsub("_val1", "", x)
    x = gsub("_val2", "", x)
    x = gsub("_test", "", x)
    return(x)
}

# Table for paper
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
    )



