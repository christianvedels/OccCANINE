# Descriptive stats for other systems
# Created:  2025-09-01
# Authors:  Christian Vedel [christian-vs@sam.sdu.dk],
#
# Purpose:  Descriptive statistics used in the paper

# YES: THIS ABSOLUTELY NEEDS TO BE CLEANED UP A LOT. SORRY!

# ==== Libraries ====
library(tidyverse)
library(foreach)
library(knitr)
library(kableExtra)
source("Project_dissemination/Paper_replication_package/Model_eval_R/000_Functions.R")

# ==== Load data ====
tl = TRUE  # Toyload for quick testing
files_train = c("EN_ISCO68_IPUMS_UK_train.csv", "EN_ISCO68_IPUMS_UK_n_10000_train.csv", "EN_OCCICEM_IPUMS_UK_train.csv", "EN_OCCICEM_IPUMS_UK_n_10000_train.csv", "EN_OCC1950_IPUMS_US_train.csv", "EN_OCC1950_IPUMS_US_n_10000_train.csv")
# files_val1 = c("EN_ISCO68_IPUMS_UK_val1.csv", "EN_ISCO68_IPUMS_UK_n_unq10000_val1.csv", "EN_OCCICEM_IPUMS_UK_val1.csv", "EN_OCCICEM_IPUMS_UK_n_unq10000_val1.csv", "EN_OCC1950_IPUMS_US_val1.csv", "EN_OCC1950_IPUMS_US_n_unq10000_val1.csv")
# files_val2 = c("EN_ISCO68_IPUMS_UK_val2.csv", "EN_ISCO68_IPUMS_UK_n_unq10000_val2.csv", "EN_OCCICEM_IPUMS_UK_val2.csv", "EN_OCCICEM_IPUMS_UK_n_unq10000_val2.csv", "EN_OCC1950_IPUMS_US_val2.csv", "EN_OCC1950_IPUMS_US_n_unq10000_val2.csv")
files_test = c("EN_ISCO68_IPUMS_UK_test.csv", "EN_OCCICEM_IPUMS_UK_test.csv", "EN_OCC1950_IPUMS_US_test.csv")

train = read0("Data/Training_data_other", verbose = TRUE, files = files_train, toyload = tl)
# val1 = read0("Data/Validation_data1_other", verbose = TRUE, files = files_val1, toyload = tl)
# val2 = read0("Data/Validation_data2_other", verbose = TRUE, files = files_val2, toyload = tl)
test = read0("Data/Test_data_other", verbose = TRUE, files = files_test, toyload = tl)

# Files with test performance
performance_files = list.files("Project_dissemination/Paper_replication_package/Data/Intermediate_data/other_systems", pattern = "_performance")
performance_files = performance_files[grepl(".csv", performance_files)]
performance = foreach(f = performance_files, .combine = rbind) %do% {
    df = read.csv(paste0("Project_dissemination/Paper_replication_package/Data/Intermediate_data/other_systems/", f))
    df$File = f
    df$System = gsub("_performance.*", "", f)
    df = df %>%
        mutate(
            Training = case_when(
                grepl("n_unq10000", File) ~ "10k unique strings",
                grepl("n_unq10000_finetuning", File) ~ "10k unique strings (full finetuning)",
                grepl("n_10000", File) ~ "10k total strings",
                grepl("n_10000_finetuning", File) ~ "10k total strings (full finetuning)",
                TRUE ~ "All data"
            ),
        )
    return(df)
}
# ==== Table: performance under training regimes ====
# Derive per-system train counts from 'train'
train_counts = train %>%
  mutate(
    SystemPretty = case_when(
      grepl("ISCO68", file, ignore.case = TRUE) ~ "ISCO-68",
      grepl("OCCICEM", file, ignore.case = TRUE) ~ "OCCICEM",
      grepl("OCC1950", file, ignore.case = TRUE) ~ "OCC1950",
      TRUE ~ NA_character_
    ),
    RegimeBase = ifelse(grepl("n_unq10000", file, ignore.case = TRUE), "10k", "full")
  ) %>%
  filter(!is.na(SystemPretty)) %>%
  group_by(SystemPretty, RegimeBase) %>%
  summarize(n_train = dplyr::n(), .groups = "drop")

get_train_n = function(sys, reg) {
  # Map table's reg to base regime in counts
  reg_base = if (reg == "full") "full" else "10k"
  val = train_counts %>% dplyr::filter(SystemPretty == sys, RegimeBase == reg_base) %>% dplyr::pull(n_train)
  if (length(val) == 0) NA_integer_ else val[1]
}

# Derive per-system test counts from 'test' (not from 'performance')
test_counts = test %>%
  mutate(
    SystemPretty = case_when(
      grepl("ISCO68", file, ignore.case = TRUE) ~ "ISCO-68",
      grepl("OCCICEM", file, ignore.case = TRUE) ~ "OCCICEM",
      grepl("OCC1950", file, ignore.case = TRUE) ~ "OCC1950",
      TRUE ~ NA_character_
    )
  ) %>%
  filter(!is.na(SystemPretty)) %>%
  group_by(SystemPretty) %>%
  summarize(test_n = dplyr::n(), .groups = "drop")

get_test_n = function(sys) {
  val = test_counts %>% dplyr::filter(SystemPretty == sys) %>% dplyr::pull(test_n)
  if (length(val) == 0) NA_integer_ else val[1]
}

# ==== Table: performance under training regimes ====
perf2 = performance %>%
  mutate(
    System = tolower(System),
    SystemPretty = dplyr::recode(System, "isco68" = "ISCO-68", "occicem" = "OCCICEM", "occ1950" = "OCC1950"),
    Regime = dplyr::case_when(
      grepl("_n_10000_full_finetuning", File) ~ "10k_total (full finetuning)", # Panel D
      grepl("_n_unq10000_full_finetuning", File) ~ "10k_full",   # Panel A
      grepl("_n_unq10000", File) ~ "10k_frozen",                 # Panel B
      grepl("_n_10000", File) ~ "10k_total",                     # Panel C
      TRUE ~ "full"                                              # Panel E
    )
  )

fmt3 = function(x) ifelse(is.na(x), NA, sprintf("%.3f", x))
co_short = function(x, decimals_m = 1) {
  if (is.na(x)) return(NA_character_)
  else scales::comma(x)  # 1,000 style (no 'k')
}

get_row = function(sys, reg) {
  r = perf2 %>% dplyr::filter(SystemPretty == sys, Regime == reg) %>% dplyr::slice(1)
  tr_n = get_train_n(sys, reg)
  te_n = get_test_n(sys)

  train_obs_str = if (is.na(tr_n)) {
    if (reg == "full") "15.8m" else "10,000"
  } else {
    if (reg == "full") co_short(tr_n) else scales::comma(tr_n)
  }
  test_obs_str = if (is.na(te_n)) {
    "931,000"
  } else {
    co_short(te_n)  # will be "1,000" style for thousands
  }

  if (nrow(r) == 0) {
    return(data.frame(
      `Target system` = sys,
      `Train obs.` = train_obs_str,
      `Test obs.` = test_obs_str,
      Accuracy = NA, Precision = NA, Recall = NA, `F1-score` = NA,
      `Train time` = dplyr::case_when(
        reg == "full" ~ "$\\sim$20 days",
        reg == "10k_full" ~ "$\\sim$48 hours",
        TRUE ~ "$\\sim$20 hours"
      ),
      check.names = FALSE
    ))
  }
  data.frame(
    `Target system` = sys,
    `Train obs.` = train_obs_str,
    `Test obs.` = test_obs_str,
    Accuracy = fmt3(r$accuracy),
    Precision = fmt3(r$precision),
    Recall = fmt3(r$recall),
    `F1-score` = fmt3(r$f1),
    `Train time` = case_when(
      reg == "full" ~ "$\\sim$20 days",
      reg == "10k_full" ~ "$\\sim$48 hours",
      reg == "10k_total (full finetuning)" ~ "$\\sim$48 hours",
      reg == "10k_frozen" ~ "$\\sim$20 hours",
      TRUE ~ "$\\sim$20 hours"
    ),
    check.names = FALSE
  )
}

panelA = dplyr::bind_rows(
  get_row("ISCO-68", "full"),
  get_row("OCCICEM", "full"),
  get_row("OCC1950", "full")
)
panelB = dplyr::bind_rows(
  get_row("ISCO-68", "10k_total (full finetuning)"),
  get_row("OCCICEM", "10k_total (full finetuning)"),
  get_row("OCC1950", "10k_total (full finetuning)")
)
panelC = dplyr::bind_rows(
  get_row("ISCO-68", "10k_total"),
  get_row("OCCICEM", "10k_total"),
  get_row("OCC1950", "10k_total")
)


tab = dplyr::bind_rows(
  panelA, 
  panelB, 
  panelC
)

sink("Project_dissemination/Paper_replication_package/Tables/Performance_other_systems.txt")
tab %>%
  kable(
    format = "latex",
    booktabs = TRUE,
    escape = FALSE,
    caption = "Model performance under different training data constraints",
    align = c("l","l","r","r","r","r","r","l")
  ) %>%
  kableExtra::pack_rows("\\textit{Panel A: Full training data (good decoder)}", 1, 3, italic = TRUE) %>%
  kableExtra::pack_rows("\\textit{Panel B: 10,000 strings, (good decoder)}", 4, 6, italic = TRUE) %>%
  kableExtra::pack_rows("\\textit{Panel C: 10,000 strings, frozen encoder (good decoder)}", 7, 9, italic = TRUE) %>%
  # kableExtra::pack_rows("\\textit{Panel B: 10,000 total strings, frozen encoder (good decoder)}", 10, 12, italic = TRUE) %>%
  # kableExtra::pack_rows("\\textit{Panel C: 10,000 total strings (good decoder)}", 13, 15, italic = TRUE) %>%
  print()
sink()

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

